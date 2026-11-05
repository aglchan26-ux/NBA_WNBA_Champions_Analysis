"""Step 5c: what roster mix goes with winning? (association, not causation)

Inputs: data/processed/team_seasons.parquet (Step 5a). Features are z-scored within league-season,
so 'more three-point diet than that season's average' is comparable across eras and leagues.

A  Regular-season net rating ~ roster mix (archetype minutes shares, baseline = Connector forward,
   plus role/shot-diet features). Ridge, mixed model (random intercept per franchise), gradient
   boosting + SHAP. Forward-chained CV: each season predicted from earlier seasons only.
   Two feature sets: 'style' (no efficiency) and 'style+efficiency' (adds efficiency vs role).
B  Playoff success beyond net rating, among playoff teams:
   B1 title: conditional logit (softmax over the season's playoff teams), L2-penalized;
      compare forward-chained log-likelihood of the actual champion: net rating only vs + roster mix.
   B2 rounds won: ridge on net rating vs net rating + roster mix, forward-chained.
   B3 playoff resilience: change in net rating from regular season to playoffs ~ roster mix.
C  Drift: B1 and A refit in rolling 8-season windows; standardized coefficients over time.
D  NBA -> WNBA: B1 fit on all NBA seasons, scored on every WNBA season, vs WNBA forward-chained.
Uncertainty: coefficient intervals by bootstrapping whole seasons (1,000 draws for ridge/logit).
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from common import MODEL_START, PROC, RESULTS

warnings.filterwarnings("ignore")
ARCH = ["On-ball creator", "Secondary guard", "Spot-up 3-and-D wing", "Hub big / point forward", "Rim-running big"]
ROLE = ["C1_primary_scorer", "C2_secondary_scorer", "C3_spacing_big_share", "C4_perimeter_d_share",
        "rim_protector_share", "n_high_usage", "usage_concentration", "team_three_diet", "team_rim_diet", "shooter_share"]
STYLE = [f"arch_share::{a}" for a in ARCH] + ROLE
EFF = ["eff_vs_role_weighted"]
QUAD = ["team_three_diet", "team_rim_diet", "usage_concentration"]  # saturation ('over-use') terms
MIN_TRAIN = 8
RNG = np.random.default_rng(42)


def load() -> pd.DataFrame:
    T = pd.read_parquet(PROC / "team_seasons.parquet").copy()
    T = T[T.season >= T.league.map(MODEL_START)].reset_index(drop=True)
    for c in ["C1_primary_scorer", "C2_secondary_scorer"]:
        T[c] = T[c].astype(float)
    for c in STYLE + EFF:
        T[c] = T[c].fillna(0.0)
        T[c + "_z"] = T.groupby(["league", "season"])[c].transform(lambda v: (v - v.mean()) / (v.std(ddof=0) or 1))
    for c in QUAD:
        T[c + "_z2"] = T[c + "_z"] ** 2
    T["net_z"] = T.groupby(["league", "season"]).net_rating.transform(lambda v: (v - v.mean()) / v.std(ddof=0))
    T["net_c"] = T.net_rating - T.groupby(["league", "season"]).net_rating.transform("mean")
    pg = pd.read_parquet(PROC / "player_game.parquet", columns=["league", "season", "season_type", "game_id", "team_id",
                                                                "poss", "t_pts", "o_pts"])
    po = pg[pg.season_type == "playoffs"].drop_duplicates(["game_id", "team_id"])
    p = po.groupby(["league", "season", "team_id"]).agg(pp=("t_pts", "sum"), op=("o_pts", "sum"), poss=("poss", "sum"))
    p["po_net"] = 100 * (p.pp - p.op) / p.poss
    T = T.merge(p.po_net.reset_index(), on=["league", "season", "team_id"], how="left")
    T["po_net_change"] = T.po_net - T.net_rating
    return T


def zcols(cols, quad=False):
    return [c + "_z" for c in cols] + ([c + "_z2" for c in QUAD] if quad else [])


# ------------------------------------------------------------------ A
def part_a(T: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    import lightgbm as lgb
    rows, coefs = [], []
    for lg, D in T.groupby("league"):
        seasons = sorted(D.season.unique())
        for fs_name, cols in [("style", zcols(STYLE, True)), ("style+efficiency", zcols(STYLE + EFF, True))]:
            preds = {"ridge": [], "gbm": [], "season_mean": []}
            ys = []
            for s in seasons[MIN_TRAIN:]:
                tr, te = D[D.season < s], D[D.season == s]
                r = Ridge(alpha=10.0).fit(tr[cols], tr.net_c)
                g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=15,
                                      subsample=0.8, subsample_freq=1, verbose=-1, random_state=1).fit(tr[cols].to_numpy(), tr.net_c)
                preds["ridge"] += list(r.predict(te[cols]))
                preds["gbm"] += list(g.predict(te[cols].to_numpy()))
                preds["season_mean"] += [0.0] * len(te)
                ys += list(te.net_c)
            ys = np.array(ys)
            for m, p in preds.items():
                p = np.array(p)
                rows.append({"league": lg, "features": fs_name, "model": m, "test_seasons": len(seasons) - MIN_TRAIN,
                             "teams": len(ys), "r2_out_of_time": 1 - ((ys - p) ** 2).sum() / ((ys - ys.mean()) ** 2).sum(),
                             "rmse": np.sqrt(((ys - p) ** 2).mean())})
            # full-sample ridge coefficients with season-bootstrap intervals (linear terms only)
            lin = zcols(STYLE + (EFF if fs_name == "style+efficiency" else []))
            base = Ridge(alpha=10.0).fit(D[lin], D.net_c).coef_
            boots = []
            for _ in range(1000):
                ss = RNG.choice(seasons, len(seasons), replace=True)
                B = pd.concat([D[D.season == s] for s in ss])
                boots.append(Ridge(alpha=10.0).fit(B[lin], B.net_c).coef_)
            boots = np.array(boots)
            for j, c in enumerate(lin):
                coefs.append({"league": lg, "features": fs_name, "term": c[:-2], "net_rating_per_sd": base[j],
                              "lo95": np.percentile(boots[:, j], 2.5), "hi95": np.percentile(boots[:, j], 97.5)})
    return pd.DataFrame(rows), pd.DataFrame(coefs)


def part_a_mixed_and_shap(T: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    import lightgbm as lgb
    import shap
    import statsmodels.formula.api as smf
    out, sh = [], []
    for lg, D in T.groupby("league"):
        D = D.copy()
        lin = zcols(STYLE)
        names = {c: f"x{i}" for i, c in enumerate(lin)}
        E = D.rename(columns=names)
        f = "net_c ~ " + " + ".join(names.values())
        try:
            m = smf.mixedlm(f, E, groups=E.team_id).fit(reml=True, method="lbfgs")
            ci = m.conf_int()
            for c, n in names.items():
                out.append({"league": lg, "term": c[:-2], "net_rating_per_sd": m.params[n], "lo95": ci.loc[n, 0],
                            "hi95": ci.loc[n, 1], "franchise_sd": float(np.sqrt(m.cov_re.iloc[0, 0]))})
        except Exception as e:
            out.append({"league": lg, "term": "mixed model failed", "note": str(e)[:80]})
        cols = zcols(STYLE, True)
        g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=15,
                              subsample=0.8, subsample_freq=1, verbose=-1, random_state=1).fit(D[cols].to_numpy(), D.net_c)
        sv = shap.TreeExplainer(g).shap_values(D[cols].to_numpy())
        for j, c in enumerate(cols):
            sh.append({"league": lg, "feature": c, "mean_abs_shap": np.abs(sv[:, j]).mean()})
    return pd.DataFrame(out), pd.DataFrame(sh).sort_values(["league", "mean_abs_shap"], ascending=[True, False])


# ------------------------------------------------------------------ B
def clogit_fit(D: pd.DataFrame, cols: list, lam: float = 1.0) -> np.ndarray:
    groups = [g for _, g in D.groupby(["league", "season"]) if g.champion.sum() == 1]
    Xs = [g[cols].to_numpy() for g in groups]
    ys = [g.champion.to_numpy().astype(bool) for g in groups]

    def nll(b):
        tot, grad = lam * b @ b, 2 * lam * b
        for X, y in zip(Xs, ys):
            u = X @ b
            u = u - u.max()
            p = np.exp(u) / np.exp(u).sum()
            tot -= np.log(p[y][0])
            grad -= X[y][0] - p @ X
        return tot, grad

    return minimize(nll, np.zeros(len(cols)), jac=True, method="L-BFGS-B").x


def clogit_ll(g: pd.DataFrame, cols: list, b: np.ndarray) -> tuple[float, float]:
    u = g[cols].to_numpy() @ b
    u = u - u.max()
    p = np.exp(u) / np.exp(u).sum()
    y = g.champion.to_numpy().astype(bool)
    return float(np.log(p[y][0])), float(p[y][0])


def part_b(T: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    P = T[T.made_playoffs].copy()
    sets = {"net rating only": ["net_z"],
            "net + checklist": ["net_z", "C1_primary_scorer_z", "C2_secondary_scorer_z", "C3_spacing_big_share_z",
                                "C4_perimeter_d_share_z"],
            "net + roster mix": ["net_z"] + zcols(STYLE),
            "net + roster mix + over-use terms": ["net_z"] + zcols(STYLE, True)}
    rows, per_season = [], []
    for lg, D in P.groupby("league"):
        seasons = sorted(D.season.unique())
        for name, cols in sets.items():
            lls, probs, rr_true, rr_pred = [], [], [], []
            for s in seasons[MIN_TRAIN:]:
                tr, te = D[D.season < s], D[D.season == s]
                if te.champion.sum() != 1:
                    continue
                b = clogit_fit(tr, cols, lam=2.0)
                ll, pc = clogit_ll(te, cols, b)
                lls.append(ll)
                probs.append(pc)
                per_season.append({"league": lg, "season": s, "model": name, "champion_prob": pc,
                                   "champion": te[te.champion].team.iat[0],
                                   "champion_rank": int((te[cols].to_numpy() @ b >= (te[te.champion][cols].to_numpy() @ b)[0]).sum())})
                r = Ridge(alpha=10.0).fit(tr[cols], tr.rounds_won)
                rr_true += list(te.rounds_won)
                rr_pred += list(r.predict(te[cols]))
            n_teams = D[D.season >= seasons[MIN_TRAIN]].groupby("season").size().mean()
            rr_true, rr_pred = np.array(rr_true), np.array(rr_pred)
            rows.append({"league": lg, "model": name, "test_seasons": len(lls), "mean_loglik_champion": np.mean(lls),
                         "uniform_baseline_loglik": -np.log(n_teams), "mean_prob_on_champion": np.mean(probs),
                         "rounds_won_rmse": np.sqrt(((rr_true - rr_pred) ** 2).mean())})
    # full-sample title model coefficients with season bootstrap
    coefs = []
    for lg, D in P.groupby("league"):
        cols = sets["net + roster mix"]
        b0 = clogit_fit(D, cols, lam=2.0)
        seasons = sorted(D.season.unique())
        bs = []
        for _ in range(300):
            ss = RNG.choice(seasons, len(seasons), replace=True)
            B = pd.concat([D[D.season == s].assign(season=1000 + i) for i, s in enumerate(ss)])
            bs.append(clogit_fit(B, cols, lam=2.0))
        bs = np.array(bs)
        for j, c in enumerate(cols):
            coefs.append({"league": lg, "term": c.replace("_z", ""), "odds_ratio_per_sd": np.exp(b0[j]),
                          "lo95": np.exp(np.percentile(bs[:, j], 2.5)), "hi95": np.exp(np.percentile(bs[:, j], 97.5))})
    return pd.DataFrame(rows), pd.DataFrame(per_season), pd.DataFrame(coefs)


def part_b3(T: pd.DataFrame) -> pd.DataFrame:
    """Playoff resilience: change in net rating, playoffs minus regular season, ~ roster mix."""
    P = T[T.made_playoffs & T.po_net_change.notna()]
    out = []
    for lg, D in P.groupby("league"):
        cols = zcols(STYLE, True) + ["net_z"]
        seasons = sorted(D.season.unique())
        y = D.po_net_change - D.groupby("season").po_net_change.transform("mean")
        base = Ridge(alpha=20.0).fit(D[cols], y).coef_
        bs = []
        for _ in range(1000):
            ss = RNG.choice(seasons, len(seasons), replace=True)
            B = pd.concat([D[D.season == s] for s in ss])
            yb = B.po_net_change - B.groupby("season").po_net_change.transform("mean")
            bs.append(Ridge(alpha=20.0).fit(B[cols], yb).coef_)
        bs = np.array(bs)
        for j, c in enumerate(cols):
            out.append({"league": lg, "term": c, "playoff_net_change_per_sd": base[j], "lo95": np.percentile(bs[:, j], 2.5),
                        "hi95": np.percentile(bs[:, j], 97.5), "teams": len(D)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ C, D
def part_c(T: pd.DataFrame) -> pd.DataFrame:
    P = T[T.made_playoffs]
    key = ["net_z", "C1_primary_scorer_z", "C3_spacing_big_share_z", "C4_perimeter_d_share_z", "rim_protector_share_z",
           "team_three_diet_z", "team_rim_diet_z", "n_high_usage_z", "usage_concentration_z"]
    out = []
    for lg, D in P.groupby("league"):
        seasons = sorted(D.season.unique())
        for i in range(0, len(seasons) - 7):
            win = seasons[i:i + 8]
            W = D[D.season.isin(win)]
            b = clogit_fit(W, key, lam=3.0)
            A = T[(T.league == lg) & T.season.isin(win)]
            r = Ridge(alpha=10.0).fit(A[[k for k in key if k != "net_z"]], A.net_c).coef_
            for j, k in enumerate(key):
                out.append({"league": lg, "window": f"{win[0]}-{win[-1]}", "mid_season": win[3], "term": k[:-2],
                            "title_logodds_per_sd": b[j], "net_rating_per_sd": (r[j - 1] if k != "net_z" else np.nan)})
    return pd.DataFrame(out)


def part_d(T: pd.DataFrame) -> pd.DataFrame:
    P = T[T.made_playoffs]
    out = []
    for name, cols in [("net rating only", ["net_z"]), ("net + roster mix", ["net_z"] + zcols(STYLE))]:
        b = clogit_fit(P[P.league == "NBA"], cols, lam=2.0)
        W = P[P.league == "WNBA"]
        lls = [clogit_ll(g, cols, b) for _, g in W.groupby("season") if g.champion.sum() == 1]
        out.append({"model": name, "trained_on": "NBA (all seasons)", "tested_on": "WNBA (every season)",
                    "seasons": len(lls), "mean_loglik_champion": np.mean([x[0] for x in lls]),
                    "mean_prob_on_champion": np.mean([x[1] for x in lls]),
                    "uniform_baseline_loglik": -np.log(W.groupby("season").size().mean())})
    return pd.DataFrame(out)


def main() -> None:
    T = load()
    T.to_parquet(PROC / "team_seasons_model.parquet", index=False)
    a, ac = part_a(T)
    a.to_csv(RESULTS / "step5_A_netrating_cv.csv", index=False)
    ac.to_csv(RESULTS / "step5_A_ridge_coefs.csv", index=False)
    am, ashap = part_a_mixed_and_shap(T)
    am.to_csv(RESULTS / "step5_A_mixed_model.csv", index=False)
    ashap.to_csv(RESULTS / "step5_A_shap_importance.csv", index=False)
    b, bps, bc = part_b(T)
    b.to_csv(RESULTS / "step5_B_title_cv.csv", index=False)
    bps.to_csv(RESULTS / "step5_B_title_by_season.csv", index=False)
    bc.to_csv(RESULTS / "step5_B_title_coefs.csv", index=False)
    b3 = part_b3(T)
    b3.to_csv(RESULTS / "step5_B3_playoff_resilience.csv", index=False)
    c = part_c(T)
    c.to_csv(RESULTS / "step5_C_drift_windows.csv", index=False)
    d = part_d(T)
    d.to_csv(RESULTS / "step5_D_nba_to_wnba.csv", index=False)
    for name, x in [("A cv", a), ("B title cv", b), ("D transfer", d)]:
        print(f"\n== {name}\n" + x.round(3).to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
