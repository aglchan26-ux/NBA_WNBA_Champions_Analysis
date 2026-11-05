"""Step 6: does pay track on-court value? Salary vs a market model, with intervals.

Scope: regular-season, counted player-seasons with a known salary. NBA 2009-10 onward (season >= 2010, where
salary and the other inputs are complete), WNBA 2021 onward (salary sheets start in 2021).

Pay scale: cap_share = salary / (league-season total salary / number of teams), i.e. salary as a share of an average
team payroll. It is a data-derived stand-in for the salary cap (no CBA figure is typed in from memory) and puts
NBA and WNBA, and every season, on one scale.

Contract cohorts (salaries in these are set by rules, not by the market):
  max_scale     top-band salary amounts shared by >= 3 players (CBA max tiers) and above: pay is capped, so these are
                reported by value delivered and never labeled over/under-paid
  NBA  rookie_scale  first-round pick in years 1-4 after the draft (from draft history)
       minimum_scale low salary amount shared by >= 5 players (minimum contracts), or prorated (<= 1.10 x 5th pct)
       veteran       everyone else (draft history ends in 2023: later first-round rookies are inferred, flagged)
  WNBA from the Her Hoop Stats signing code: rookie_scale, development, short_term -> rule-set; core_max_tier,
       standard -> veteran; unknown / Spotrac fill -> veteran, flagged.

Market model ("fair pay"): gradient boosting on log(cap_share), trained ONLY on veteran contracts, cross-fitted by
season (a player's own season is never in the training fold). Features (no salary information):
  this season: player impact (offense, defense; lineup-based), minutes, availability (games / team games), age, age^2,
  position, archetype; last season: impact, minutes (contracts are signed on past performance).
Interval: quantile gradient boosting (5th and 95th percentile of log cap_share) gives a 90% band for fair pay.

Surplus: log(fair) - log(actual); positive = underpaid. Never a division by salary.
Low-salary shrinkage: s_shrunk = s * actual / (actual + lambda); lambda chosen by cross-validation (folds by
  season) so that shrunk surplus best predicts the change in a player's log cap share the next season among players
  whose pay changed (a new contract), i.e. the surplus the market later corrects.
Label: underpaid / overpaid only when the 90% interval for surplus excludes 0; otherwise 'fairly paid (uncertain)'.
Availability: surplus is recomputed with availability = 1 (a fully healthy season); the difference is the part of
  any overpay due to games missed.
Old vs new: the notebook's rule (predicted / actual > 1.2 undervalued, < 0.8 overvalued; no interval) applied to the
  same predictions, counted next to the new rule.
Outputs: data/processed/salary_value.parquet, results/step6_{counts,cohort_summary,model_cv,top_players,lambda}.csv
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from common import PROC, RESULTS

warnings.filterwarnings("ignore")
START = {"NBA": 2010, "WNBA": 2021}
LAMBDAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]


def load() -> pd.DataFrame:
    S = pd.read_parquet(PROC / "player_season_salary.parquet")
    S = S[S.salary.notna() & (S.salary > 0)]
    S = S[S.season >= S.league.map(START)]
    keep = ["league", "season", "athlete_id", "player", "position", "age", "minutes", "games", "salary",
            "rookie_scale", "draft_year", "cohort", "contract_code", "salary_source"]
    S = S[[c for c in keep if c in S.columns]].copy()
    part = pd.read_parquet(PROC / "participation.parquet",
                           columns=["league", "season", "athlete_id", "counted", "availability", "likely_season_ending"])
    S = S.merge(part, on=["league", "season", "athlete_id"], how="left")
    S = S[S.counted.fillna(False).astype(bool)]
    P = pd.read_parquet(PROC / "player_impact.parquet", columns=["league", "season", "athlete_id", "o_impact", "d_impact", "impact"])
    S = S.merge(P, on=["league", "season", "athlete_id"], how="left")
    prev = P.assign(season=P.season + 1).rename(columns={"o_impact": "o_prev", "d_impact": "d_prev", "impact": "impact_prev"})
    S = S.merge(prev, on=["league", "season", "athlete_id"], how="left")
    ps = pd.read_parquet(PROC / "player_season.parquet")
    ps = ps[ps.season_type == "regular"][["league", "season", "athlete_id", "minutes"]].rename(columns={"minutes": "minutes_prev"})
    S = S.merge(ps.assign(season=ps.season + 1), on=["league", "season", "athlete_id"], how="left")
    A = pd.read_parquet(PROC / "archetypes_detailed.parquet", columns=["league", "season", "athlete_id", "archetype"])
    S = S.merge(A, on=["league", "season", "athlete_id"], how="left")
    pay = S.groupby(["league", "season"]).salary.transform("sum")
    teams = pd.read_parquet(PROC / "team_seasons.parquet").groupby(["league", "season"]).team_id.nunique()
    S["n_teams"] = S.set_index(["league", "season"]).index.map(teams)
    S["cap_share"] = S.salary / (pay / S.n_teams)
    # cohorts
    # minimum-scale (data-derived): minimum contracts are paid in identical amounts to many players, so a salary at or
    # below the league-season 35th percentile that is shared by >= 5 players is a minimum-scale amount; salaries at or
    # below 1.10 x the 5th percentile are prorated / partial-season deals and join the same cohort.
    amt = S.salary.round(0)
    shared = S.groupby(["league", "season", amt]).salary.transform("size") >= 5
    p35 = S.groupby(["league", "season"]).salary.transform(lambda s: s.quantile(0.35))
    p5 = S.groupby(["league", "season"]).salary.transform(lambda s: s.quantile(0.05))
    minimum = (shared & (S.salary <= p35)) | (S.salary <= 1.10 * p5)
    # max-scale: max contracts are also identical amounts (tiers set by the CBA); top-band amounts (>= 90th pct)
    # shared by >= 3 players, and anything at or above the smallest such amount that season
    p90 = S.groupby(["league", "season"]).salary.transform(lambda s: s.quantile(0.90))
    shared3 = S.groupby(["league", "season", amt]).salary.transform("size") >= 3
    floor = S[shared3 & (S.salary >= p90)].groupby(["league", "season"]).salary.min()
    fl = pd.Series(list(zip(S.league, S.season))).map(floor).to_numpy()
    maxs = (S.salary >= np.where(np.isnan(fl.astype(float)), np.inf, fl.astype(float))).to_numpy()
    nba = S.league == "NBA"
    # rookie scale: draft history (first-round, years 1-4) where it exists (drafts to 2023); for later debuts with no
    # draft record, a non-minimum salary in the first 4 seasons is treated as rookie scale and flagged as inferred
    debut = pd.read_parquet(PROC / "player_season.parquet", columns=["league", "season", "athlete_id"]).groupby(
        ["league", "athlete_id"]).season.min().rename("debut")
    S = S.join(debut, on=["league", "athlete_id"])
    S["minimum"] = minimum.to_numpy()
    rk_draft = S.rookie_scale.fillna(False).astype(bool)
    rk_inferred = S.draft_year.isna() & (S.debut >= 2025) & (S.season - S.debut <= 3) & ~S.minimum if "draft_year" in S else False
    S["cohort6"] = "veteran"
    S.loc[nba & S.minimum, "cohort6"] = "minimum_scale"
    S.loc[nba & (rk_draft | rk_inferred) & ~S.minimum, "cohort6"] = "rookie_scale"
    S.loc[maxs & ~(nba & (rk_draft | rk_inferred)), "cohort6"] = "max_scale"
    wc = S.cohort.fillna("unknown")
    rule = wc.isin(["rookie_scale", "development", "short_term"])
    S.loc[~nba & rule, "cohort6"] = wc
    S.loc[~nba & ~rule & S.minimum, "cohort6"] = "minimum_scale"
    S.loc[~nba & ~rule & maxs, "cohort6"] = "max_scale"
    S["cohort_flag"] = np.select([nba & rk_inferred & ~S.minimum, ~nba & (wc == "unknown")], ["rookie_scale_inferred", "cohort_unknown"], "")
    return S.reset_index(drop=True)


FEATS = ["o_impact", "d_impact", "minutes", "availability", "age", "age2", "o_prev", "d_prev", "minutes_prev",
         "pos_G", "pos_C"] + [f"arch::{a}" for a in ["On-ball creator", "Secondary guard", "Spot-up 3-and-D wing",
                                                       "Connector forward", "Hub big / point forward", "Rim-running big"]]


# more impact, minutes, availability (this or last season) never lowers fair pay
MONO = [1 if f in {"o_impact", "d_impact", "minutes", "availability", "o_prev", "d_prev", "minutes_prev"} else 0 for f in FEATS]


def features(S: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=S.index)
    for c in ["o_impact", "d_impact", "minutes", "availability", "age", "o_prev", "d_prev", "minutes_prev"]:
        X[c] = S[c]
    X["age2"] = (S.age - 27) ** 2
    X["pos_G"] = (S.position == "G").astype(float)
    X["pos_C"] = (S.position == "C").astype(float)
    for a in FEATS[11:]:
        X[a] = (S.archetype == a.split("::")[1]).astype(float)
    return X[FEATS]


def cross_fit(S: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, vet: np.ndarray, healthy: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=S.index, columns=["pred", "lo", "hi", "pred_healthy"], dtype=float)
    for lg in S.league.unique():
        L = (S.league == lg).to_numpy()
        for s in sorted(S.season[L].unique()):
            te = L & (S.season == s).to_numpy()
            tr = L & vet & (S.season != s).to_numpy()
            mods = {k: HistGradientBoostingRegressor(loss=l, quantile=q, max_iter=300, learning_rate=0.05,
                                                     min_samples_leaf=20, random_state=0, monotonic_cst=MONO).fit(X[tr], y[tr])
                    for k, l, q in (("pred", "squared_error", None), ("lo", "quantile", 0.05), ("hi", "quantile", 0.95))}
            for k, m in mods.items():
                out.loc[te, k] = m.predict(X[te])
            out.loc[te, "pred_healthy"] = mods["pred"].predict(healthy[te])
    return out


def main() -> None:
    S = load()
    X = features(S)
    healthy = X.copy()
    healthy["availability"] = 1.0
    healthy["minutes"] = S.minutes / S.availability.clip(lower=0.15)       # same minutes per game, every game played
    y = np.log(S.cap_share.to_numpy())
    vet = (S.cohort6 == "veteran").to_numpy()
    F = cross_fit(S, X, y, vet, healthy)
    S["fair_log"] = F.pred
    lo, hi = np.minimum(F.lo, F.pred), np.maximum(F.hi, F.pred)
    # conformalized quantile regression: widen the band by the 90th percentile of out-of-band errors of
    # veterans in OTHER seasons (same league), so the band covers ~90% out of season
    E = pd.Series(np.maximum(lo - y, y - hi), index=S.index)
    adj = pd.Series(0.0, index=S.index)
    for (lg, s_), g in S.groupby(["league", "season"]):
        other = vet & (S.league == lg).to_numpy() & (S.season != s_).to_numpy()
        adj[g.index] = np.quantile(E[other], 0.90)
    S["fair_lo"], S["fair_hi"] = lo - adj, hi + adj
    S["surplus"] = S.fair_log - y
    S["surplus_lo"], S["surplus_hi"] = S.fair_lo - y, S.fair_hi - y
    S["surplus_healthy"] = F.pred_healthy - y
    S["availability_part"] = S.surplus - S.surplus_healthy       # < 0: pay lost to games missed

    # market-model accuracy on veterans (out-of-season)
    cv = []
    for lg, g in S[vet].groupby("league"):
        r = g.fair_log - np.log(g.cap_share)
        cover = ((np.log(g.cap_share) >= g.fair_lo) & (np.log(g.cap_share) <= g.fair_hi)).mean()
        cv.append({"league": lg, "veteran_contracts": len(g),
                   "r2_log_pay": 1 - (r ** 2).sum() / ((np.log(g.cap_share) - np.log(g.cap_share).mean()) ** 2).sum(),
                   "interval_coverage_90": cover})
    pd.DataFrame(cv).to_csv(RESULTS / "step6_model_cv.csv", index=False)

    # lambda for low-salary shrinkage: shrunk surplus vs next-season change in log cap share (new contracts)
    nxt = S[["league", "season", "athlete_id", "cap_share"]].assign(season=S.season - 1).rename(columns={"cap_share": "cap_next"})
    J = S.merge(nxt, on=["league", "season", "athlete_id"])
    J["dlog"] = np.log(J.cap_next) - np.log(J.cap_share)
    J = J[J.dlog.abs() > 0.15]                                       # pay changed: a new contract
    lam_rows = []
    for lg, g in J.groupby("league"):
        seasons = sorted(g.season.unique())
        for lam in LAMBDAS:
            err = []
            for s in seasons:                                        # leave-one-season-out: fit slope on others
                tr, te = g[g.season != s], g[g.season == s]
                xs = lambda d: d.surplus * d.cap_share / (d.cap_share + lam)
                b = np.polyfit(xs(tr), tr.dlog, 1)
                err += list((np.polyval(b, xs(te)) - te.dlog) ** 2)
            lam_rows.append({"league": lg, "lambda": lam, "cv_mse": np.mean(err), "n": len(g)})
    LAM = pd.DataFrame(lam_rows)
    LAM.to_csv(RESULTS / "step6_lambda.csv", index=False)
    best = LAM.loc[LAM.groupby("league").cv_mse.idxmin()].set_index("league")["lambda"]
    lam = S.league.map(best)
    w = S.cap_share / (S.cap_share + lam)
    S["surplus_shrunk"], S["surplus_lo_s"], S["surplus_hi_s"] = S.surplus * w, S.surplus_lo * w, S.surplus_hi * w
    S["label"] = np.select([S.cohort6 == "max_scale", S.surplus_lo_s > 0, S.surplus_hi_s < 0],
                           ["max contract (pay capped: not labeled)", "underpaid", "overpaid"], "fairly paid (uncertain)")
    ratio = np.exp(S.fair_log) / S.cap_share
    S["label_old_ratio"] = np.select([ratio > 1.2, ratio < 0.8], ["underpaid", "overpaid"], "fairly paid")
    S["overpaid_mainly_availability"] = (S.label == "overpaid") & (S.availability_part < 0) & (S.surplus_healthy > S.surplus_hi_s)

    counts = []
    for (lg, coh), g in S.groupby(["league", "cohort6"]):
        counts.append({"league": lg, "cohort": coh, "player_seasons": len(g),
                       "old_rule_underpaid": (g.label_old_ratio == "underpaid").mean(),
                       "old_rule_overpaid": (g.label_old_ratio == "overpaid").mean(),
                       "new_rule_underpaid": (g.label == "underpaid").mean(),
                       "new_rule_overpaid": (g.label == "overpaid").mean(),
                       "median_surplus_shrunk": g.surplus_shrunk.median()})
    C = pd.DataFrame(counts)
    C.to_csv(RESULTS / "step6_counts.csv", index=False)
    S.to_parquet(PROC / "salary_value.parquet", index=False)

    last = S.groupby("league").season.transform("max")
    top = S[(S.season == last) & (S.label != "fairly paid (uncertain)")].copy()
    top["salary_m"] = (top.salary / 1e6).round(2)
    cols = ["league", "season", "player", "cohort6", "salary_m", "impact", "surplus_shrunk", "surplus_lo_s", "surplus_hi_s",
            "label", "availability_part"]
    mk = top.cohort6.isin(["veteran"])
    tp = pd.concat([top[mk & (top.label == "underpaid")].sort_values("surplus_shrunk", ascending=False).groupby("league").head(10),
                    top[mk & (top.label == "overpaid")].sort_values("surplus_shrunk").groupby("league").head(10)])
    tp[cols].round(3).to_csv(RESULTS / "step6_top_players.csv", index=False)
    mx = S[(S.season == last) & (S.cohort6 == "max_scale")].copy()
    mx["salary_m"] = (mx.salary / 1e6).round(2)
    mx = mx.sort_values(["league", "impact"], ascending=[True, False])
    mx[["league", "season", "player", "salary_m", "o_impact", "d_impact", "impact", "minutes", "availability"]].round(2).to_csv(
        RESULTS / "step6_max_contracts.csv", index=False)
    print(pd.DataFrame(cv).round(3).to_string(index=False), file=sys.stderr)
    print("lambda:", best.to_dict(), file=sys.stderr)
    print(C.round(3).to_string(index=False), file=sys.stderr)
    print(tp[cols].round(2).to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
