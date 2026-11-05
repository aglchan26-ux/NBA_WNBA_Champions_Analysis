"""Title models with player impact from lineups, and archetype mix x player quality.

Leakage rule: a player's impact estimated from THIS season's games, summed over a roster, largely rebuilds
this season's team net rating. So the roster-based models here use each player's impact from the PREVIOUS
season (prior-informed RAPM, player_impact.py), applied to this season's roster and minutes. Nothing in these
features comes from this season's team results. Players with no previous-season estimate count as average (0).

Team features (minutes-weighted, x5 = per lineup; z-scored within league-season):
  impact_prev        sum of 5 * share * impact(s-1)            (overall roster quality)
  o_prev, d_prev     same for offense and defense separately
  q::<archetype>     5 * sum over that archetype's players of share * impact(s-1)
                     (archetype mix x player quality: how much GOOD wing / big / creator play the team has)
  avail_*            the same, using only players available for the playoffs (absent_playoffs removed),
                     with shares renormalized
  plus archetype shares, age, and the star / efficiency features from step5_title_no_net.py
Same forward-chained conditional-logit test as step5_models / step5_title_no_net (playoff teams, MIN_TRAIN seasons).
Also: a playoff-SERIES model (Bradley-Terry style logit on feature differences) using every series, not only
champions, scored forward-chained by log-loss and accuracy.
Outputs: results/step5_title_impact_cv.csv, step5_series_cv.csv, step5_title_impact_coefs.csv
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import PROC, RESULTS
from step5_models import MIN_TRAIN, STYLE, clogit_fit, clogit_ll, load
from step5_title_no_net import roster_extras

warnings.filterwarnings("ignore")
ARCH = ["On-ball creator", "Secondary guard", "Spot-up 3-and-D wing", "Connector forward", "Hub big / point forward", "Rim-running big"]


def impact_features() -> pd.DataFrame:
    P = pd.read_parquet(PROC / "player_impact.parquet")[["league", "season", "athlete_id", "impact", "o_impact", "d_impact"]]
    prev = P.assign(season=P.season + 1)
    pt = pd.read_parquet(PROC / "player_team_season.parquet")
    pt = pt[pt.season_type == "regular"][["league", "season", "athlete_id", "team_id", "minutes"]]
    A = pd.read_parquet(PROC / "archetypes_detailed.parquet", columns=["league", "season", "athlete_id", "archetype"])
    part = pd.read_parquet(PROC / "participation.parquet", columns=["league", "season", "athlete_id", "absent_playoffs"])
    d = pt.merge(prev, how="left").merge(A, how="left").merge(part, how="left")
    d[["impact", "o_impact", "d_impact"]] = d[["impact", "o_impact", "d_impact"]].fillna(0)
    d["absent_playoffs"] = d.absent_playoffs.fillna(False).astype(bool)

    def team(g: pd.DataFrame, prefix: str) -> pd.Series:
        s = 5 * g.minutes / g.minutes.sum()
        out = {f"{prefix}impact_prev": (s * g.impact).sum(), f"{prefix}o_prev": (s * g.o_impact).sum(),
               f"{prefix}d_prev": (s * g.d_impact).sum(),
               f"{prefix}top3_prev": g.nlargest(3, "minutes").impact.mean()}
        for a in ARCH:
            m = g.archetype == a
            out[f"{prefix}q::{a}"] = (s[m] * g.impact[m]).sum()
        return pd.Series(out)

    allp = d.groupby(["league", "season", "team_id"]).apply(lambda g: team(g, ""))
    avail = d[~d.absent_playoffs].groupby(["league", "season", "team_id"]).apply(lambda g: team(g, "avail_"))
    return allp.join(avail).reset_index()


def zscore(T: pd.DataFrame, cols: list) -> list:
    out = []
    for c in cols:
        T[c] = T[c].fillna(0.0).astype(float)
        T[c + "_z"] = T.groupby(["league", "season"])[c].transform(lambda v: (v - v.mean()) / (v.std(ddof=0) or 1))
        out.append(c + "_z")
    return out


def title_cv(T: pd.DataFrame, sets: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, per = [], []
    for lg, D in T[T.made_playoffs].groupby("league"):
        seasons = sorted(D.season.unique())
        seasons = [s for s in seasons if s > D.season.min()]  # need a previous season for impact
        for name, cols in sets.items():
            lls, probs, ranks = [], [], []
            for s in seasons[MIN_TRAIN:]:
                tr, te = D[(D.season < s) & (D.season > D.season.min())], D[D.season == s]
                if te.champion.sum() != 1:
                    continue
                b = clogit_fit(tr, cols, lam=2.0)
                ll, pc = clogit_ll(te, cols, b)
                u = te[cols].to_numpy() @ b
                rank = int((u >= u[te.champion.to_numpy().astype(bool)][0]).sum())
                lls.append(ll), probs.append(pc), ranks.append(rank)
                per.append({"league": lg, "season": s, "model": name, "champion": te[te.champion].team.iat[0],
                            "champion_prob": pc, "champion_rank": rank})
            n = D[D.season >= seasons[MIN_TRAIN]].groupby("season").size().mean()
            rows.append({"league": lg, "model": name, "test_seasons": len(lls), "mean_loglik_champion": np.mean(lls),
                         "random_pick_loglik": -np.log(n), "mean_prob_on_champion": np.mean(probs),
                         "champion_ranked_1st": np.mean(np.array(ranks) == 1), "champion_in_top3": np.mean(np.array(ranks) <= 3)})
    return pd.DataFrame(rows), pd.DataFrame(per)


def series_table() -> pd.DataFrame:
    """Playoff series from playoff games: pair of teams per league-season, winner = more wins in the series."""
    pg = pd.read_parquet(PROC / "player_game.parquet", columns=["league", "season", "season_type", "game_id", "game_date",
                                                                "team_id", "opponent_team_id", "t_pts", "o_pts", "game_context"])
    g = pg[(pg.season_type == "playoffs")].drop_duplicates(["game_id", "team_id"])
    g = g[g.team_id < g.opponent_team_id].copy()
    g["a_win"] = g.t_pts > g.o_pts
    s = g.groupby(["league", "season", "team_id", "opponent_team_id"]).agg(games=("game_id", "size"), a_wins=("a_win", "sum"),
                                                                           start=("game_date", "min")).reset_index()
    s = s[s.a_wins * 2 != s.games]
    s["a_won"] = (s.a_wins * 2 > s.games).astype(int)
    return s.rename(columns={"team_id": "team_a", "opponent_team_id": "team_b"})


PER_SERIES = []


def series_cv(T: pd.DataFrame, sets: dict) -> pd.DataFrame:
    S = series_table()
    rows = []
    for lg, D in S.groupby("league"):
        TT = T[T.league == lg].set_index(["season", "team_id"])
        seasons = sorted(D.season.unique())
        seasons = [s for s in seasons if s > T[T.league == lg].season.min()]
        for name, cols in sets.items():
            Xa = TT.reindex(pd.MultiIndex.from_arrays([D.season, D.team_a]))[cols].to_numpy()
            Xb = TT.reindex(pd.MultiIndex.from_arrays([D.season, D.team_b]))[cols].to_numpy()
            X = np.nan_to_num(Xa - Xb)
            y = D.a_won.to_numpy()
            # symmetric: add flipped copies so the model has no 'team a' bias
            ll, acc, n = [], [], 0
            for s in seasons[MIN_TRAIN:]:
                tr = (D.season < s).to_numpy() & (D.season > seasons[0] - 1).to_numpy()
                te = (D.season == s).to_numpy()
                if te.sum() == 0:
                    continue
                Xt, yt = np.r_[X[tr], -X[tr]], np.r_[y[tr], 1 - y[tr]]
                m = LogisticRegression(C=0.5, fit_intercept=False).fit(Xt, yt)
                p = np.clip(m.predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
                ll += list(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p))
                PER_SERIES.append(pd.DataFrame({"league": lg, "model": name, "season": s, "i": np.where(te)[0],
                                                "ll": y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)}))
                acc += list((p > 0.5) == y[te])
            rows.append({"league": lg, "model": name, "series_tested": len(ll), "mean_logloss": -np.mean(ll),
                         "coin_flip_logloss": np.log(2), "accuracy": np.mean(acc)})
    return pd.DataFrame(rows)


def paired_bootstrap(ps: pd.DataFrame, n: int = 2000) -> pd.DataFrame:
    """Gain in series log-likelihood vs 'net rating only', resampling whole seasons (2,000 draws)."""
    rng = np.random.default_rng(11)
    rows = []
    for lg, g in ps.groupby("league"):
        w = g.pivot_table(index=["season", "i"], columns="model", values="ll")
        base = w["net rating only (reference)"]
        seasons = w.index.get_level_values(0).unique().to_numpy()
        for m in w.columns:
            if m == "net rating only (reference)":
                continue
            diff = (w[m] - base).groupby(level=0).sum()
            cnt = w[m].groupby(level=0).size()
            boots = []
            for _ in range(n):
                ss = rng.choice(seasons, len(seasons))
                boots.append(diff.loc[ss].sum() / cnt.loc[ss].sum())
            est = diff.sum() / cnt.sum()
            rows.append({"league": lg, "model": m, "loglik_gain_per_series": est,
                         "lo95": np.percentile(boots, 2.5), "hi95": np.percentile(boots, 97.5),
                         "p_gain_le_0": float(np.mean(np.array(boots) <= 0))})
    return pd.DataFrame(rows)


def main() -> None:
    T = load()
    T = T.merge(roster_extras(T), on=["league", "season", "team_id"], how="left")
    T = T.merge(impact_features(), on=["league", "season", "team_id"], how="left")
    base = zscore(T, ["eff_w", "eff_top3", "stars", "age_w", "avail_eff_w", "avail_stars", "avail_minutes_lost"])
    imp = zscore(T, ["impact_prev", "o_prev", "d_prev", "top3_prev"])
    q = zscore(T, [f"q::{a}" for a in ARCH])
    av = zscore(T, ["avail_impact_prev", "avail_o_prev", "avail_d_prev", "avail_top3_prev"])
    aq = zscore(T, [f"avail_q::{a}" for a in ARCH])
    mix = [c + "_z" for c in STYLE]
    sets = {
        "net rating only (reference)": ["net_z"],
        "player impact (prev season): overall": ["impact_prev_z"],
        "player impact (prev season): offense + defense": ["o_prev_z", "d_prev_z"],
        "archetype mix x player quality": q,
        "playoff-available: offense + defense": ["avail_o_prev_z", "avail_d_prev_z"],
        "playoff-available: archetype x quality": aq,
        "playoff-available: impact + archetype x quality + age": ["avail_o_prev_z", "avail_d_prev_z"] + aq + ["age_w_z"],
        "playoff-available: impact + mix + age": ["avail_o_prev_z", "avail_d_prev_z", "age_w_z"] + mix,
        "net rating + playoff-available impact": ["net_z", "avail_o_prev_z", "avail_d_prev_z"],
    }
    cv, per = title_cv(T, sets)
    cv.to_csv(RESULTS / "step5_title_impact_cv.csv", index=False)
    per.to_csv(RESULTS / "step5_title_impact_by_season.csv", index=False)
    print(cv.round(3).to_string(index=False), file=sys.stderr)
    T["net_rating_raw"] = T.net_rating
    sc = series_cv(T, {k: v for k, v in sets.items()})
    sc.to_csv(RESULTS / "step5_series_cv.csv", index=False)
    sig = paired_bootstrap(pd.concat(PER_SERIES))
    sig.to_csv(RESULTS / "step5_series_significance.csv", index=False)
    print(sig.round(3).to_string(index=False), file=sys.stderr)
    print(sc.round(3).to_string(index=False), file=sys.stderr)
    T.drop(columns=[c for c in T.columns if c.endswith("_z")]).to_parquet(PROC / "team_seasons_impact.parquet", index=False)


if __name__ == "__main__":
    main()
