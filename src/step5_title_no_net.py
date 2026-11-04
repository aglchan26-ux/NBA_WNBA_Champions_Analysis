"""Title models WITHOUT net rating, win % or team ratings: roster composition, archetypes and player stats only.

Question: can who is on the roster (and how they play) pick champions as well as, or better than, the team's
own results (net rating)? All models are forward-chained (train on seasons before s, predict season s),
among playoff teams, with a conditional logit (one champion per league-season, L2 penalty).

Feature blocks (each z-scored within league-season):
  mix        archetype minutes shares + role checklist (from step5_models.STYLE)
  quality    player efficiency vs role (minutes-weighted), top-3 players' efficiency, star count
  stats      team shot diet, rim-block and steal rates, roster age vs league
  available  the same mix / quality computed only on players who were available for the playoffs
             (players flagged absent_playoffs in participation.py removed)
  two-stage  composition -> predicted net rating (ridge trained on all teams of past seasons) -> title
Reference: net rating only (from step5_models).
Outputs: results/step5_title_no_net_cv.csv, step5_title_no_net_by_season.csv
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from common import PROC, RESULTS
from step5_models import MIN_TRAIN, STYLE, clogit_fit, clogit_ll, load

warnings.filterwarnings("ignore")


def roster_extras(T: pd.DataFrame) -> pd.DataFrame:
    """Player-level aggregates per team-season: quality, age, and playoff-available versions."""
    pt = pd.read_parquet(PROC / "player_team_season.parquet")
    pt = pt[pt.season_type == "regular"][["league", "season", "athlete_id", "team_id", "minutes"]]
    L = pd.read_parquet(PROC / "player_layers.parquet", columns=[
        "league", "season", "athlete_id", "archetype", "subtype", "alpha_scorer", "eff_vs_role"])
    A = pd.read_parquet(PROC / "player_season_salary.parquet", columns=["league", "season", "athlete_id", "age"])
    P = pd.read_parquet(PROC / "participation.parquet", columns=["league", "season", "athlete_id", "absent_playoffs", "counted"])
    d = pt.merge(L, how="left").merge(A, how="left").merge(P, how="left")
    d["eff_vs_role"] = d.eff_vs_role.fillna(0)
    d["star"] = (d.alpha_scorer.fillna(False).astype(bool) | (d.subtype == "Shot-making scorer")).astype(float)
    d["absent_playoffs"] = d.absent_playoffs.fillna(False).astype(bool)

    def agg(g: pd.DataFrame) -> pd.Series:
        w = g.minutes
        top = g.nlargest(3, "minutes")
        out = {"eff_w": np.average(g.eff_vs_role, weights=w), "eff_top3": np.average(top.eff_vs_role, weights=top.minutes),
               "stars": g.star[g.minutes >= 1000].sum(),
               "age_w": np.average(g.age.fillna(g.age.mean() if g.age.notna().any() else 27), weights=w)}
        av = g[~g.absent_playoffs]
        aw = av.minutes
        out["avail_minutes_lost"] = 1 - aw.sum() / w.sum()
        out["avail_eff_w"] = np.average(av.eff_vs_role, weights=aw) if aw.sum() else 0.0
        out["avail_stars"] = av.star[av.minutes >= 1000].sum()
        for a in ["On-ball creator", "Secondary guard", "Spot-up 3-and-D wing", "Hub big / point forward", "Rim-running big"]:
            out[f"avail_arch::{a}"] = aw[av.archetype == a].sum() / aw.sum() if aw.sum() else 0.0
        return pd.Series(out)

    X = d.groupby(["league", "season", "team_id"]).apply(agg).reset_index()
    return X


def main() -> None:
    T = load()
    T = T.merge(roster_extras(T), on=["league", "season", "team_id"], how="left")
    extra = ["eff_w", "eff_top3", "stars", "age_w", "avail_minutes_lost", "avail_eff_w", "avail_stars"] + \
            [c for c in T.columns if c.startswith("avail_arch::")] + ["def_rim_block_z", "def_steal_z"]
    for c in extra:
        T[c] = T[c].fillna(0.0)
        T[c + "_z"] = T.groupby(["league", "season"])[c].transform(lambda v: (v - v.mean()) / (v.std(ddof=0) or 1))
    mix = [c + "_z" for c in STYLE]
    quality = ["eff_w_z", "eff_top3_z", "stars_z"]
    stats = ["def_rim_block_z_z", "def_steal_z_z", "age_w_z"]  # shot diet is already in mix (STYLE)
    avail = [c + "_z" for c in T.columns if c.startswith("avail_arch::") and not c.endswith("_z")] + \
            ["avail_eff_w_z", "avail_stars_z", "avail_minutes_lost_z"]
    sets = {
        "net rating only (reference)": ["net_z"],
        "archetype mix": mix,
        "player quality (efficiency vs role, stars)": quality,
        "mix + quality": mix + quality,
        "mix + quality + stats (age, shot diet, defense)": mix + quality + stats,
        "playoff-available roster: mix + quality + stats": avail + stats,
        "two-stage: composition -> predicted net -> title": ["pred_net_z"],
    }
    comp_for_net = mix + quality + stats
    rows, per = [], []
    for lg, D_all in T.groupby("league"):
        seasons = sorted(D_all.season.unique())
        for s in seasons[MIN_TRAIN:]:
            tr_all = D_all[D_all.season < s]
            r = Ridge(alpha=10.0).fit(tr_all[comp_for_net], tr_all.net_z)
            D_all.loc[D_all.season <= s, "pred_net_tmp"] = r.predict(D_all.loc[D_all.season <= s, comp_for_net])
            T.loc[D_all.index[D_all.season == s], "pred_net"] = D_all.loc[D_all.season == s, "pred_net_tmp"]
        # in-sample predicted net for training seasons: refit per test season below
    T["pred_net"] = T.pred_net.fillna(0.0)
    for lg, D_all in T.groupby("league"):
        P = D_all[D_all.made_playoffs].copy()
        seasons = sorted(P.season.unique())
        for name, cols in sets.items():
            lls, probs, ranks = [], [], []
            for s in seasons[MIN_TRAIN:]:
                tr, te = P[P.season < s].copy(), P[P.season == s].copy()
                if te.champion.sum() != 1:
                    continue
                if name.startswith("two-stage"):
                    # predicted net for training rows must also come from a model that did not see them:
                    # use the ridge fit on seasons before s (training rows are then in-sample for that ridge,
                    # which only flatters the training fit, never the test season).
                    tall = D_all[D_all.season < s]
                    rr = Ridge(alpha=10.0).fit(tall[comp_for_net], tall.net_z)
                    for X in (tr, te):
                        X["pred_net_z"] = rr.predict(X[comp_for_net])
                b = clogit_fit(tr, cols, lam=2.0)
                ll, pc = clogit_ll(te, cols, b)
                u = te[cols].to_numpy() @ b
                rank = int((u >= u[te.champion.to_numpy().astype(bool)][0]).sum())
                lls.append(ll), probs.append(pc), ranks.append(rank)
                per.append({"league": lg, "season": s, "model": name, "champion": te[te.champion].team.iat[0],
                            "champion_prob": pc, "champion_rank": rank, "playoff_teams": len(te)})
            n = P[P.season >= seasons[MIN_TRAIN]].groupby("season").size().mean()
            rows.append({"league": lg, "model": name, "test_seasons": len(lls), "mean_loglik_champion": np.mean(lls),
                         "random_pick_loglik": -np.log(n), "mean_prob_on_champion": np.mean(probs),
                         "champion_ranked_1st": np.mean(np.array(ranks) == 1),
                         "champion_in_top3": np.mean(np.array(ranks) <= 3)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "step5_title_no_net_cv.csv", index=False)
    pd.DataFrame(per).to_csv(RESULTS / "step5_title_no_net_by_season.csv", index=False)
    print(out.round(3).to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
