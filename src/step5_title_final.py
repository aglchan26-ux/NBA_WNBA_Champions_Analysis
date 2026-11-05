"""Final title / series model: net rating plus our roster variables, trained as a time series.

New variable, availability-adjusted strength:
  avail_now   sum over players AVAILABLE for the playoffs of 5 x minutes share x this season's player impact
              (prior-informed lineup RAPM). It is known when the regular season ends (before the playoffs), so it is
              a fair predictor. Net rating counts every player who played; avail_now drops the ones who will miss
              the playoffs (absent_playoffs in participation.py) and re-weights the rest.
  avail_o_now, avail_d_now  the same, offense and defense separately.
  Also used: archetype x quality (available players), roster age.
Star features: avail_star = impact of the best available player (1,000+ minutes); avail_top2 = top two.
Penalty: the L2 strength (2, 10, 30) is chosen in-chain together with the half-life.
Time-series training: forward-chained (season s predicted from seasons < s). Training seasons get weight
  0.5 ** ((s - 1 - season) / half_life): recent seasons count more as the game drifts. The half-life is chosen
  inside the forward chain (each test season uses the half-life that did best on the seasons before it), so the
  choice never sees the test season.
Models compared (title: conditional logit over playoff teams; series: logit on team differences):
  net rating only | availability-adjusted strength only | net + availability-adjusted strength |
  net + availability-adjusted (offense, defense) + archetype x quality + age
Weights: standardized coefficients of the full-sample fit, so the weight of net rating can be compared with ours.
Outputs: results/step5_final_title_cv.csv, step5_final_series_cv.csv, step5_final_significance.csv, step5_final_weights.csv
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from common import PROC, RESULTS
from step5_models import MIN_TRAIN, load
from step5_title_impact import ARCH, paired_bootstrap, series_table, zscore

warnings.filterwarnings("ignore")
HALF_LIVES = [None, 12, 8, 5]
LAMS = [2.0, 10.0, 30.0]          # L2 penalty: chosen in-chain, like the half-life
GRID = [(h, l) for h in HALF_LIVES for l in LAMS]


def avail_now() -> pd.DataFrame:
    P = pd.read_parquet(PROC / "player_impact.parquet")[["league", "season", "athlete_id", "impact", "o_impact", "d_impact"]]
    pt = pd.read_parquet(PROC / "player_team_season.parquet")
    pt = pt[pt.season_type == "regular"][["league", "season", "athlete_id", "team_id", "minutes"]]
    A = pd.read_parquet(PROC / "archetypes_detailed.parquet", columns=["league", "season", "athlete_id", "archetype"])
    part = pd.read_parquet(PROC / "participation.parquet", columns=["league", "season", "athlete_id", "absent_playoffs"])
    d = pt.merge(P, how="left").merge(A, how="left").merge(part, how="left")
    d[["impact", "o_impact", "d_impact"]] = d[["impact", "o_impact", "d_impact"]].fillna(0)
    d = d[~d.absent_playoffs.fillna(False).astype(bool)]

    def team(g):
        s = 5 * g.minutes / g.minutes.sum()
        big = g[g.minutes >= 1000].impact.sort_values(ascending=False)
        out = {"avail_now": (s * g.impact).sum(), "avail_o_now": (s * g.o_impact).sum(), "avail_d_now": (s * g.d_impact).sum(),
               "avail_star": big.iloc[0] if len(big) else 0.0,                 # best available player (1,000+ min)
               "avail_top2": big.iloc[:2].sum() if len(big) else 0.0}
        for a in ARCH:
            m = g.archetype == a
            out[f"availq_now::{a}"] = (s[m] * g.impact[m]).sum()
        return pd.Series(out)
    return d.groupby(["league", "season", "team_id"]).apply(team).reset_index()


def wclogit_fit(D, cols, w_season, lam=2.0):
    groups = [(g[cols].to_numpy(), g.champion.to_numpy().astype(bool), w_season.get(s, 1.0))
              for s, g in D.groupby("season") if g.champion.sum() == 1]

    def nll(b):
        tot, grad = lam * b @ b, 2 * lam * b
        for X, y, w in groups:
            u = X @ b
            u = u - u.max()
            p = np.exp(u) / np.exp(u).sum()
            tot -= w * np.log(p[y][0])
            grad -= w * (X[y][0] - p @ X)
        return tot, grad
    return minimize(nll, np.zeros(len(cols)), jac=True, method="L-BFGS-B").x


def weights(seasons, s, hl):
    return {x: (1.0 if hl is None else 0.5 ** ((s - 1 - x) / hl)) for x in seasons}


def title_ll(te, cols, b):
    u = te[cols].to_numpy() @ b
    u = u - u.max()
    p = np.exp(u) / np.exp(u).sum()
    y = te.champion.to_numpy().astype(bool)
    return float(np.log(p[y][0])), float(p[y][0]), int((u >= u[y][0]).sum())


def run_title(T, sets):
    rows, per = [], []
    for lg, D in T[T.made_playoffs].groupby("league"):
        seasons = sorted(D.season.unique())
        for name, cols in sets.items():
            res, hist = [], {g: [] for g in GRID}
            for s in seasons[MIN_TRAIN:]:
                tr, te = D[D.season < s], D[D.season == s]
                if te.champion.sum() != 1:
                    continue
                # choose (half-life, penalty) from past test seasons only (default: no decay, lam 2)
                pick = max(GRID, key=lambda g: np.mean(hist[g]) if hist[g] else (0 if g == (None, 2.0) else -1))
                fits = {}
                for g in GRID:
                    bg = wclogit_fit(tr, cols, weights(seasons, s, g[0]), lam=g[1])
                    fits[g] = bg
                    hist[g].append(title_ll(te, cols, bg)[0])
                b = fits[pick]
                hl = pick
                ll, pc, rank = title_ll(te, cols, b)
                res.append((ll, pc, rank))
                per.append({"league": lg, "season": s, "model": name, "half_life": hl[0], "lam": hl[1], "champion_prob": pc, "champion_rank": rank})
            r = np.array(res)
            rows.append({"league": lg, "model": name, "test_seasons": len(r), "mean_loglik_champion": r[:, 0].mean(),
                         "mean_prob_on_champion": r[:, 1].mean(), "champion_ranked_1st": (r[:, 2] == 1).mean(),
                         "champion_in_top3": (r[:, 2] <= 3).mean()})
    return pd.DataFrame(rows), pd.DataFrame(per)


PER_SERIES = []


def run_series(T, sets):
    S = series_table()
    rows = []
    for lg, D in S.groupby("league"):
        D = D[D.season >= T[T.league == lg].season.min()]
        TT = T[T.league == lg].set_index(["season", "team_id"])
        seasons = sorted(D.season.unique())
        for name, cols in sets.items():
            Xa = TT.reindex(pd.MultiIndex.from_arrays([D.season, D.team_a]))[cols].to_numpy()
            Xb = TT.reindex(pd.MultiIndex.from_arrays([D.season, D.team_b]))[cols].to_numpy()
            X, y, ss = np.nan_to_num(Xa - Xb), D.a_won.to_numpy(), D.season.to_numpy()
            ll, acc = [], []
            hist = {g: [] for g in GRID}
            for s in seasons[MIN_TRAIN:]:
                tr, te = ss < s, ss == s
                if te.sum() == 0:
                    continue
                hl = max(GRID, key=lambda g: np.mean(hist[g]) if hist[g] else (0 if g == (None, 2.0) else -1))
                preds = {}
                for g in GRID:
                    w = np.array([1.0 if g[0] is None else 0.5 ** ((s - 1 - x) / g[0]) for x in ss[tr]])
                    m = LogisticRegression(C=1.0 / g[1], fit_intercept=False).fit(np.r_[X[tr], -X[tr]], np.r_[y[tr], 1 - y[tr]],
                                                                                 sample_weight=np.r_[w, w])
                    p = np.clip(m.predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
                    preds[g] = p
                    hist[g].append(np.mean(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)))
                p = preds[hl]
                l = y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)
                ll += list(l)
                acc += list((p > 0.5) == y[te])
                PER_SERIES.append(pd.DataFrame({"league": lg, "model": name, "season": s, "i": np.where(te)[0], "ll": l}))
            rows.append({"league": lg, "model": name, "series_tested": len(ll), "mean_logloss": -np.mean(ll), "accuracy": np.mean(acc)})
    return pd.DataFrame(rows)


def main():
    T = load()
    T = T.merge(avail_now(), on=["league", "season", "team_id"], how="left")
    ours = zscore(T, ["avail_now", "avail_o_now", "avail_d_now", "avail_star", "avail_top2"])
    aq = zscore(T, [f"availq_now::{a}" for a in ARCH])
    from step5_title_no_net import roster_extras
    T = T.merge(roster_extras(T)[["league", "season", "team_id", "age_w"]], on=["league", "season", "team_id"], how="left")
    age = zscore(T, ["age_w"])
    sets = {
        "net rating only (reference)": ["net_z"],
        "availability-adjusted strength only": ["avail_now_z"],
        "net rating + availability-adjusted strength": ["net_z", "avail_now_z"],
        "net + availability-adjusted strength + star": ["net_z", "avail_now_z", "avail_star_z"],
        "net + availability-adjusted strength + star + archetype x quality":
            ["net_z", "avail_now_z", "avail_star_z"] + aq,
        "net + availability-adjusted (offense, defense) + archetype x quality + age":
            ["net_z", "avail_o_now_z", "avail_d_now_z"] + aq + age,
    }
    tcv, per = run_title(T, sets)
    tcv.to_csv(RESULTS / "step5_final_title_cv.csv", index=False)
    per.to_csv(RESULTS / "step5_final_title_by_season.csv", index=False)
    scv = run_series(T, sets)
    scv.to_csv(RESULTS / "step5_final_series_cv.csv", index=False)
    sig = paired_bootstrap(pd.concat(PER_SERIES))
    sig.to_csv(RESULTS / "step5_final_significance.csv", index=False)
    # full-sample standardized weights (title model, no decay) with season bootstrap
    rng = np.random.default_rng(9)
    wrows = []
    for lg, D in T[T.made_playoffs].groupby("league"):
        cols = sets["net rating + availability-adjusted strength"]
        b0 = wclogit_fit(D, cols, {})
        seasons = D.season.unique()
        bs = []
        for _ in range(300):
            pick = rng.choice(seasons, len(seasons))
            B = pd.concat([D[D.season == x].assign(season=10000 + i) for i, x in enumerate(pick)])
            bs.append(wclogit_fit(B, cols, {}))
        bs = np.array(bs)
        for j, c in enumerate(cols):
            wrows.append({"league": lg, "variable": c.replace("_z", ""), "weight_per_sd": b0[j],
                          "lo95": np.percentile(bs[:, j], 2.5), "hi95": np.percentile(bs[:, j], 97.5),
                          "share_of_total_weight": abs(b0[j]) / np.abs(b0).sum()})
    W = pd.DataFrame(wrows)
    W.to_csv(RESULTS / "step5_final_weights.csv", index=False)
    print(tcv.round(3).to_string(index=False), file=sys.stderr)
    print(scv.round(3).to_string(index=False), file=sys.stderr)
    print(sig.round(3).to_string(index=False), file=sys.stderr)
    print(W.round(3).to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
