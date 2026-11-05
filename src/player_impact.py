"""Player impact from lineups: regularized adjusted plus-minus (RAPM), offense and defense, per league-season.

Model (regular season stints from build_lineups.py, one row per stint and offense side):
    100 * pts / poss  =  mean + home + sum(offense effects of the 5 attackers) - sum(defense effects of the 5 defenders)
weighted by possessions, with a ridge (L2) penalty that pulls every effect toward its prior:
  * single-season: prior 0 (classic RAPM)
  * prior-informed: prior = the player's previous-season effect (0 for a new player); fitted as ridge on
    the residual y - X @ prior, so the estimate is prior + this season's evidence.
Positive o_impact = more points scored per 100 possessions; positive d_impact = fewer points allowed.
impact = o_impact + d_impact (points per 100 possessions vs an average player, other nine held fixed).

Penalty strength: chosen per league by 5-fold cross-validation over games (folds keep whole games together)
on three reference seasons, then used for every season of that league.

Validation (results/player_impact_validation.csv):
  A. within season: fit on the first half of games (by date), predict the second half's stint scoring;
     weighted R^2 vs predicting the league mean.
  B. next season: team net rating in season s predicted from players' prior-informed impact in s-1,
     weighted by each player's minutes share in s; compared with last season's team net rating.
Outputs: data/processed/player_impact.parquet, results/player_impact_{alpha,validation}.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from common import PROC, RESULTS, SEASONS

ST = PROC / "stints"
ALPHAS = [250, 500, 1000, 2000, 4000, 8000]


def load(lg: str, y: int, season_type: str = "regular") -> pd.DataFrame:
    d = pd.read_parquet(ST / f"{lg}_{y}.parquet")
    d = d[(d.season_type == season_type) & (d.poss > 0)].reset_index(drop=True)
    return d


def design(d: pd.DataFrame, players: np.ndarray | None = None):
    off = d[[f"off_{k}" for k in range(1, 6)]].to_numpy()
    de = d[[f"def_{k}" for k in range(1, 6)]].to_numpy()
    if players is None:
        players = np.unique(np.r_[off.ravel(), de.ravel()])
    pos = {p: i for i, p in enumerate(players)}
    n, m = len(d), len(players)
    rows = np.repeat(np.arange(n), 5)
    oc = np.array([pos.get(p, -1) for p in off.ravel()])
    dc = np.array([pos.get(p, -1) for p in de.ravel()])
    ko, kd = oc >= 0, dc >= 0
    X = sparse.hstack([
        sparse.csr_matrix((np.ones(ko.sum()), (rows[ko], oc[ko])), shape=(n, m)),
        sparse.csr_matrix((-np.ones(kd.sum()), (rows[kd], dc[kd])), shape=(n, m)),   # minus: + d effect = fewer points
        sparse.csr_matrix(d[["home"]].to_numpy().astype(float) - 0.5),
    ]).tocsr()
    y = 100 * d.pts.to_numpy() / d.poss.to_numpy()
    w = d.poss.to_numpy()
    return X, y, w, players


def fit(X, y, w, alpha, prior=None):
    mu = np.average(y, weights=w)
    r = y - mu - (X @ prior if prior is not None else 0)
    m = Ridge(alpha=alpha, fit_intercept=False, solver="sparse_cg", max_iter=2000, tol=1e-4).fit(X, r, sample_weight=w)
    b = m.coef_ + (prior if prior is not None else 0)
    return b, mu


def wr2(y, yhat, w, base):
    return 1 - np.sum(w * (y - yhat) ** 2) / np.sum(w * (y - base) ** 2)


def choose_alpha(lg: str) -> pd.DataFrame:
    yrs = list(SEASONS[lg.lower()])
    ref = [yrs[len(yrs) // 4], yrs[len(yrs) // 2], yrs[-2]]
    rows = []
    for y in ref:
        d = load(lg, y)
        X, yy, w, _ = design(d)
        games = d.game_id.unique()
        fold = pd.Series(np.random.default_rng(0).integers(0, 5, len(games)), index=games)[d.game_id].to_numpy()
        for a in ALPHAS:
            num = den = 0.0
            for k in range(5):
                tr, te = fold != k, fold == k
                b, mu = fit(X[tr], yy[tr], w[tr], a)
                p = mu + X[te] @ b
                num += np.sum(w[te] * (yy[te] - p) ** 2)
                den += np.sum(w[te] * (yy[te] - np.average(yy[tr], weights=w[tr])) ** 2)
            rows.append({"league": lg, "season": y, "alpha": a, "cv_r2": 1 - num / den})
    return pd.DataFrame(rows)


BOX = ["usage_eb", "pts_eb", "ast_eb", "orb_eb", "drb_eb", "stl_eb", "blk_eb", "tov_eb", "pf_eb", "fta_rate_eb",
       "fg3_rate_eb", "assisted_eb", "fg3_pct_eb", "ft_pct_eb", "rim_pct_eb", "mid_pct_eb"]
# prior = BOX_W * box-score prior + PREV_W * last season's estimate (prev decays toward 0 when PREV_W < 1)
CONFIGS = {"A prev x1.0 (old)": (0.0, 1.0), "B prev x0.5": (0.0, 0.5), "C box-score prior": (1.0, 0.0),
           "D box 0.5 + prev 0.5": (0.5, 0.5)}


def box_prior(single: pd.DataFrame) -> pd.DataFrame:
    """Box-score prior: ridge of single-season RAPM (offense, defense) on shrunk box-score rates (z within
    league-season, plus scoring per usage), fitted leave-one-season-out so a season's prior never uses its own RAPM."""
    d = pd.read_parquet(PROC / "player_season_shrunk.parquet")
    d = d[d.season_type == "regular"][["league", "season", "athlete_id"] + BOX].copy()
    d["eff"] = d.pts_eb / d.usage_eb.clip(lower=1e-3)
    feats = BOX + ["eff"]
    for c in feats:
        d[c] = d.groupby(["league", "season"])[c].transform(lambda v: (v - v.mean()) / (v.std(ddof=0) or 1))
    d = d.merge(single, on=["league", "season", "athlete_id"], how="left")
    out = []
    for lg, g in d.groupby("league"):
        for s_ in g.season.unique():
            tr = g[(g.season != s_) & g.o_single.notna() & (g.poss >= 1000)]
            te = g[g.season == s_]
            X = tr[feats].fillna(0).to_numpy()
            pr = {}
            for side in ("o", "d"):
                m = Ridge(alpha=50.0).fit(X, tr[f"{side}_single"], sample_weight=tr.poss)
                pr[f"{side}_box"] = m.predict(te[feats].fillna(0).to_numpy())
            out.append(pd.DataFrame({"league": lg, "season": s_, "athlete_id": te.athlete_id.to_numpy(), **pr}))
    return pd.concat(out, ignore_index=True)


def holdout_r2(X, yy, w, first, alpha, prior):
    bh, muh = fit(X[first], yy[first], w[first], alpha, prior=prior)
    p = muh + X[~first] @ bh
    return wr2(yy[~first], p, w[~first], muh)


def main() -> None:
    gd = pd.read_parquet(PROC / "player_game.parquet", columns=["league", "season", "game_id", "game_date"]).drop_duplicates("game_id")
    data, alph, single_rows = {}, [], []
    for lg in ["NBA", "WNBA"]:
        A = choose_alpha(lg)
        alph.append(A)
        alpha = int(A.groupby("alpha").cv_r2.mean().idxmax())
        print(f"{lg}: alpha {alpha}", file=sys.stderr)
        for y in SEASONS[lg.lower()]:
            d = load(lg, y)
            X, yy, w, players = design(d)
            b0, _ = fit(X, yy, w, alpha)
            m = len(players)
            offs = d[[f"off_{k}" for k in range(1, 6)]].to_numpy().ravel()
            poss = pd.Series(np.repeat(d.poss.to_numpy(), 5)).groupby(offs).sum().reindex(players).fillna(0).to_numpy()
            dates = pd.to_datetime(gd.set_index("game_id").game_date.reindex(d.game_id)).astype("int64").to_numpy()
            first = dates <= np.median(dates)
            data[(lg, y)] = (d, X, yy, w, players, alpha, first)
            single_rows.append(pd.DataFrame({"league": lg, "season": y, "athlete_id": players, "o_single": b0[:m],
                                             "d_single": b0[m:2 * m], "poss": poss}))
    single = pd.concat(single_rows, ignore_index=True)
    BP = box_prior(single[["league", "season", "athlete_id", "o_single", "d_single", "poss"]])

    results, val = {}, []
    for name, (bw, pw) in CONFIGS.items():
        est = []
        for lg in ["NBA", "WNBA"]:
            prev = None
            for y in SEASONS[lg.lower()]:
                d, X, yy, w, players, alpha, first = data[(lg, y)]
                m = len(players)
                bp = BP[(BP.league == lg) & (BP.season == y)].set_index("athlete_id")
                prior = np.zeros(X.shape[1])
                for side, sl in (("o", slice(0, m)), ("d", slice(m, 2 * m))):
                    box = bp[f"{side}_box"].reindex(players).fillna(0).to_numpy()
                    pv = prev.set_index("athlete_id")[f"{side}_impact"].reindex(players).fillna(0).to_numpy() if prev is not None else 0
                    prior[sl] = bw * box + pw * pv
                b1, _ = fit(X, yy, w, alpha, prior=prior)
                res = pd.DataFrame({"league": lg, "season": y, "athlete_id": players, "o_impact": b1[:m], "d_impact": b1[m:2 * m]})
                est.append(res)
                val.append({"league": lg, "season": y, "test": "A within-season holdout", "model": name,
                            "r2": holdout_r2(X, yy, w, first, alpha, prior)})
                prev = res
        results[name] = pd.concat(est, ignore_index=True)
    V = pd.DataFrame(val)
    score = V.groupby(["league", "model"]).r2.mean().unstack(0)
    print(score.round(5).to_string(), file=sys.stderr)
    best = score.mean(1).idxmax()          # one method for both leagues: best average holdout fit
    print("chosen:", best, file=sys.stderr)
    P = results[best].merge(single.rename(columns={"o_single": "o_impact_single", "d_single": "d_impact_single", "poss": "poss_off"}),
                            on=["league", "season", "athlete_id"], how="left")
    P["impact"] = P.o_impact + P.d_impact
    P["impact_single"] = P.o_impact_single + P.d_impact_single
    P["method"] = best
    P.to_parquet(PROC / "player_impact.parquet", index=False)
    pd.concat(alph).to_csv(RESULTS / "player_impact_alpha.csv", index=False)
    score.reset_index().to_csv(RESULTS / "player_impact_prior_comparison.csv", index=False)
    nxt = []
    for name, R in results.items():
        R = R.assign(impact=R.o_impact + R.d_impact, impact_single=0.0)
        t = next_season_test(R)
        t = t[t.model.str.startswith("prior-season player impact (prior")].assign(model=f"prior-season player impact: {name}")
        nxt.append(t)
    V = pd.concat([V[V.model == best].assign(model="prior-informed"), next_season_test(P), pd.concat(nxt)], ignore_index=True)
    V.to_csv(RESULTS / "player_impact_validation.csv", index=False)
    print(V[V.test.str.startswith("B")].round(3).to_string(index=False), file=sys.stderr)


def next_season_test(P: pd.DataFrame) -> pd.DataFrame:
    """Team net rating in s from players' s-1 impact, weighted by their minutes share in s (who actually played).
    Baseline: the team's own net rating in s-1. Rookies / players with no s-1 estimate count as 0 (average)."""
    pt = pd.read_parquet(PROC / "player_team_season.parquet")
    pt = pt[pt.season_type == "regular"][["league", "season", "athlete_id", "team_id", "minutes"]]
    ts = pd.read_parquet(PROC / "team_seasons.parquet")[["league", "season", "team_id", "net_rating"]]
    prevP = P.assign(season=P.season + 1)[["league", "season", "athlete_id", "impact", "impact_single"]]
    d = pt.merge(prevP, how="left", on=["league", "season", "athlete_id"]).fillna({"impact": 0, "impact_single": 0})
    d["share"] = d.minutes / d.groupby(["league", "season", "team_id"]).minutes.transform("sum")
    # 5 players on the floor: sum of shares = 1 per team -> multiply by 5 for "lineup" impact
    team = d.assign(r=5 * d.share * d.impact, s=5 * d.share * d.impact_single).groupby(["league", "season", "team_id"])[["r", "s"]].sum().reset_index()
    t = team.merge(ts, on=["league", "season", "team_id"])
    lastyr = ts.assign(season=ts.season + 1).rename(columns={"net_rating": "net_prev"})
    t = t.merge(lastyr, on=["league", "season", "team_id"], how="left")
    rows = []
    for lg, g in t.groupby("league"):
        g = g[g.season > g.season.min()].dropna(subset=["net_prev"])
        for nm, x in (("prior-season player impact (prior-informed)", g.r), ("prior-season player impact (single-season)", g.s),
                      ("last season team net rating", g.net_prev)):
            rows.append({"league": lg, "season": -1, "test": "B next-season team net rating", "model": nm,
                         "r2": float(np.corrcoef(x, g.net_rating)[0, 1] ** 2)})
        # both together (does player impact add to last year's team result?)
        Xm = np.c_[np.ones(len(g)), g.r, g.net_prev]
        beta = np.linalg.lstsq(Xm, g.net_rating, rcond=None)[0]
        rows.append({"league": lg, "season": -1, "test": "B next-season team net rating",
                     "model": "player impact + last season net rating (in-sample)",
                     "r2": wr2(g.net_rating.to_numpy(), Xm @ beta, np.ones(len(g)), g.net_rating.mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
