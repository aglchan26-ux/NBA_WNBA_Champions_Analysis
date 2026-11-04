"""Does fitting archetypes only on regular-rotation players improve them?

Fit set variants (everyone is still ASSIGNED; the filter only decides who shapes the clusters):
  all            current model (minutes-weighted, no cutoff)
  g15_m5         played >= 15% of team games AND >= 5% of available team minutes
  g20_m5         played >= 20% of team games AND >= 5% of available team minutes
Available minutes = team games x game length (48 NBA / 40 WNBA). Cluster names are carried
over from the current model by one-to-one centroid matching. Checked against: bootstrap
stability, hand-label agreement (batch 1 = tuning, batch 2 = untouched for this choice), and
agreement with the current labels.
"""
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from common import PROC, RESULTS
from step3_archetypes import model_matrix, weighted_resample

SEED = 7


def main() -> None:
    S = pd.read_parquet(PROC / "style_matrix.parquet")
    ps = pd.read_parquet(PROC / "player_season.parquet", columns=["league", "season", "season_type", "athlete_id", "games"])
    ps = ps[ps.season_type == "regular"].drop(columns="season_type")
    pg = pd.read_parquet(PROC / "player_game.parquet", columns=["league", "season", "season_type", "team_id", "game_id"])
    tg = pg[pg.season_type == "regular"].groupby(["league", "season", "team_id"]).game_id.nunique()
    team_games = tg.groupby(["league", "season"]).median().rename("team_games")
    S = S.merge(ps, on=["league", "season", "athlete_id"]).join(team_games, on=["league", "season"])
    glen = np.where(S.league == "NBA", 48.0, 40.0)
    S["games_share"] = S.games / S.team_games
    S["minutes_share"] = S.minutes / (S.team_games * glen)
    X, w = model_matrix(S), S.minutes.to_numpy()
    with open(PROC / "archetype_models.pkl", "rb") as fh:
        M = pickle.load(fh)
    cur = pd.Series(M["kmeans"].predict(X)).map(M["names"]).to_numpy()
    variants = {"all": np.ones(len(S), bool),
                "g15_m5": ((S.games_share >= .15) & (S.minutes_share >= .05)).to_numpy(),
                "g20_m5": ((S.games_share >= .20) & (S.minutes_share >= .05)).to_numpy()}
    hands = {b: pd.read_csv(RESULTS / f"step3_hand_label_comparison_{b}.csv") for b in ["v1", "v2"]}
    rows = []
    for name, keep in variants.items():
        km = KMeans(6, n_init=20, random_state=SEED).fit(X[keep], sample_weight=w[keep])
        cost = ((km.cluster_centers_[:, None, :] - M["kmeans"].cluster_centers_[None, :, :]) ** 2).sum(-1)
        r, c = linear_sum_assignment(cost)
        names = {int(i): M["names"][int(j)] for i, j in zip(r, c)}
        lab = pd.Series(km.predict(X)).map(names).to_numpy()
        rng = np.random.default_rng(SEED)
        Xk, wk = X[keep], w[keep]
        ref = km.predict(Xk)
        stab = np.mean([adjusted_rand_score(ref, KMeans(6, n_init=4, random_state=b).fit(
            Xk[weighted_resample(len(Xk), wk, rng)]).predict(Xk)) for b in range(6)])
        row = {"variant": name, "fit_player_seasons": int(keep.sum()), "fit_share_of_minutes": w[keep].sum() / w.sum(),
               "stability": stab, "agreement_with_current": (lab == cur).mean()}
        S["lab"] = lab
        for b, h in hands.items():
            h = h[h.hand_archetype.notna() & (h.hand_archetype != "Blended / tweener")]
            m = h.merge(S[["league", "season", "player", "lab"]], left_on=["league", "season_num", "player"],
                        right_on=["league", "season", "player"])
            for lg, g in m.groupby("league"):
                row[f"hand_{b}_{lg}"] = (g.lab == g.hand_archetype).mean()
            row[f"hand_{b}_all"] = (m.lab == m.hand_archetype).mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "step3_participation_filter_test.csv", index=False)
    print(out.round(3).to_string(index=False), file=sys.stderr)
    print("excluded from fit (g15_m5) share of player-seasons:", round(1 - variants["g15_m5"].mean(), 3), file=sys.stderr)


if __name__ == "__main__":
    main()
