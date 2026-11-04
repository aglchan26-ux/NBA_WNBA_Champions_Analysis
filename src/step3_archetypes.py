"""Step 3a: archetypes. Baseline replication, then style clustering on shrunk features.

Baseline (notebook): K-Means on per-game / quality features for 2024-25 NBA players with
>=10 games and >=10 minutes per game. PER, BPM and VORP are not computable from these box
scores without league adjustments, so the replication uses the remaining features.

New: style-only features (no efficiency, no volume), shrunk in Step 2:
  usage, ast, orb, drb, tov, stl, blk, fta_rate, assisted share, and shot-zone shares
  rim / short_mid / long_mid / corner3 (above-the-break 3 is the remainder; including it
  would make the shares collinear).
Transform: logit for shares, log for rates, then z-score within league-season so a role is
measured relative to its own league and era (NBA and WNBA can share one model).
Weighting: by minutes (K-Means sample weights; Gaussian mixtures fit on minutes-weighted
resamples) - no minutes cutoff, but a 4-minute player barely moves the fit.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from participation import fit_weight
from common import PROC, RESULTS

RATES = ["usage", "ast", "orb", "drb", "tov", "stl", "blk", "fta_rate"]
SHARES = ["assisted", "rim_share", "short_mid_share", "long_mid_share", "corner3_share"]
FEATURES = RATES + SHARES
K_RANGE = range(2, 13)
# Listed position (G=-1, F=0, C=+1) enters the final model as one extra column, scaled so a
# full G-to-C difference weighs like ~sqrt(13) feature-SDs x POS_WEIGHT/2. POS_WEIGHT=1.0 was the
# largest weight that kept 6 distinct, stable archetypes (stability 0.95). It was chosen while
# looking at hand-label batch 1, so batch 1 is a tuning set; batch 2 is the untouched check.
POS_WEIGHT = 1.0
POS_CODE = {"G": -1.0, "F": 0.0, "C": 1.0}
N_BOOT = 10
SEED = 7


def raw_style(d: pd.DataFrame) -> pd.DataFrame:
    """log / logit transforms of shrunk features (before league-season z-scoring)."""
    x = pd.DataFrame(index=d.index)
    for f in RATES:
        x[f] = np.log(d[f"{f}_eb"].clip(lower=1e-6))
    for f in SHARES:
        p = d[f"{f}_eb"].clip(1e-4, 1 - 1e-4)
        x[f] = np.log(p / (1 - p))
    return x


def style_norms(d: pd.DataFrame) -> pd.DataFrame:
    x = raw_style(d)
    g = x.groupby([d.league, d.season])
    return pd.concat({"mean": g.mean(), "sd": g.std(ddof=0)}, axis=1)


def style_matrix(d: pd.DataFrame, norms: pd.DataFrame | None = None) -> pd.DataFrame:
    """League-season z-scores. With `norms`, use saved player-season norms (for team stints)."""
    x = raw_style(d)
    if norms is None:
        norms = style_norms(d)
    idx = pd.MultiIndex.from_arrays([d.league, d.season])
    mu = norms["mean"].reindex(idx).to_numpy()
    sd = norms["sd"].reindex(idx).to_numpy()
    return pd.DataFrame((x.to_numpy() - mu) / sd, index=d.index, columns=x.columns)


def model_matrix(S: pd.DataFrame) -> np.ndarray:
    pos = S.position.map(POS_CODE).fillna(0.0).to_numpy()
    return np.column_stack([S[FEATURES].to_numpy(), POS_WEIGHT * pos * np.sqrt(len(FEATURES)) / 2])


def weighted_resample(n: int, w: np.ndarray, rng) -> np.ndarray:
    return rng.choice(len(w), size=n, replace=True, p=w / w.sum())


def select_k(X: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    sil_idx = weighted_resample(4000, w, rng)
    rows = []
    for k in K_RANGE:
        km = KMeans(k, n_init=10, random_state=SEED).fit(X, sample_weight=w)
        boot_idx = weighted_resample(len(X), w, rng)
        gm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=SEED).fit(X[boot_idx])
        ref_km, ref_gm = km.predict(X), gm.predict(X)
        ari_km, ari_gm = [], []
        for b in range(N_BOOT):
            idx = weighted_resample(len(X), w, rng)
            ari_km.append(adjusted_rand_score(ref_km, KMeans(k, n_init=4, random_state=b).fit(X[idx]).predict(X)))
            ari_gm.append(adjusted_rand_score(ref_gm, GaussianMixture(k, covariance_type="full", n_init=1,
                                                                      random_state=b).fit(X[idx]).predict(X)))
        rows.append({"k": k, "kmeans_silhouette": silhouette_score(X[sil_idx], km.predict(X[sil_idx])),
                     "kmeans_stability_ari": np.mean(ari_km), "kmeans_stability_sd": np.std(ari_km),
                     "gmm_bic": gm.bic(X[boot_idx]), "gmm_stability_ari": np.mean(ari_gm),
                     "gmm_stability_sd": np.std(ari_gm)})
        print(f"k={k} done", file=sys.stderr)
    return pd.DataFrame(rows)


def baseline_replication() -> pd.DataFrame:
    """Notebook-style K-Means on 2024-25 NBA per-game features."""
    ps = pd.read_parquet(PROC / "player_season.parquet")
    d = ps[(ps.league == "NBA") & (ps.season == 2025) & (ps.season_type == "regular")].copy()
    d = d[(d.games >= 10) & (d.minutes / d.games >= 10)]
    g = d.games
    f = pd.DataFrame({
        "PTS": d.pts / g, "RPG": (d.orb + d.drb) / g, "APG": d.ast / g, "SPG": d.stl / g, "BPG": d.blk / g,
        "FG%": d.fgm / d.fga, "3P%": d.fg3m / d.fg3a, "FT%": d.ftm / d.fta,
        "TS%": d.pts / (2 * (d.fga + 0.44 * d.fta)), "USG": (d.fga + 0.44 * d.fta + d.tov) / d.poss_on,
        "AST_TOV": d.ast / d.tov.clip(lower=1), "PTS_per_min": d.pts / d.minutes,
        "REB_per_min": (d.orb + d.drb) / d.minutes})
    f = f.fillna(f.median())
    X = StandardScaler().fit_transform(f)
    rows = []
    for k in range(2, 8):
        km = KMeans(k, random_state=42, n_init=10).fit(X)
        lab = pd.Series(km.labels_, index=d.index)
        mpg = d.minutes / d.games
        eta2 = mpg.groupby(lab).transform("mean").var() / mpg.var()  # share of minutes-per-game variance between clusters
        rows.append({"k": k, "n_players": len(d), "silhouette": silhouette_score(X, km.labels_),
                     "minutes_per_game_eta2": eta2, "sizes": "|".join(map(str, np.bincount(km.labels_)))})
    return pd.DataFrame(rows)


def main() -> None:
    base = baseline_replication()
    base.to_csv(RESULTS / "step3_baseline_kmeans.csv", index=False)
    d = pd.read_parquet(PROC / "player_season_shrunk.parquet")
    d = d[d.minutes > 0].reset_index(drop=True)
    norms = style_norms(d)
    norms.to_parquet(PROC / "style_norms.parquet")
    Z = style_matrix(d, norms)
    X, w = Z.to_numpy(), d.minutes.to_numpy()
    sel = select_k(X, w)
    sel.to_csv(RESULTS / "step3_k_selection.csv", index=False)
    pd.concat([d[["league", "season", "athlete_id", "player", "position", "minutes"]], Z], axis=1).to_parquet(
        PROC / "style_matrix.parquet", index=False)
    print(base.round(3).to_string(index=False), file=sys.stderr)
    print(sel.round(3).to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()


def name_clusters(prof: pd.DataFrame) -> dict:
    """Deterministic naming from profile signs, so labels survive refits with different cluster ids."""
    names = {}
    for c, r in prof.iterrows():
        if r.usage > 0.8 and r.ast > 0.8 and r.assisted < -0.8:
            names[c] = "On-ball creator"
        elif r.usage > 0.5 and (r.drb > 0.6 or r.orb > 0.6):
            names[c] = "Hub big / point forward"
        elif r.rim_share > 1.0 and r.blk > 0.8:
            names[c] = "Rim-running big"
        elif r.usage < -0.5 and r.assisted > 0.8 and r.corner3_share > 0.5:
            names[c] = "Spot-up 3-and-D wing"
        elif r.ast > 0.3 and r.drb < -0.5:
            names[c] = "Secondary guard"
        else:
            names[c] = "Connector forward"
    return names


def fit_final(k: int = 6) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common import FIGS
    S = pd.read_parquet(PROC / "style_matrix.parquet")
    X, w = model_matrix(S), fit_weight(S)  # participation rule: see participation.py
    km = KMeans(k, n_init=20, random_state=SEED).fit(X, sample_weight=w)
    S["cluster"] = km.predict(X)
    rng = np.random.default_rng(SEED)
    gm = GaussianMixture(k, covariance_type="full", n_init=5, random_state=SEED).fit(
        X[weighted_resample(len(X), w, rng)])
    prob = gm.predict_proba(X)
    # GMM components are numbered independently of K-Means clusters: match them by centroid.
    from scipy.optimize import linear_sum_assignment
    cost = ((km.cluster_centers_[:, None, :] - gm.means_[None, :, :]) ** 2).sum(-1)
    km_idx, gm_idx = linear_sum_assignment(cost)
    comp_of = dict(zip(km_idx, gm_idx))
    prob = prob[:, [comp_of[c] for c in range(k)]]  # column c = K-Means cluster c
    S["gmm_cluster"] = prob.argmax(1)
    S["gmm_max_prob"] = prob.max(1)
    S["own_membership"] = prob[np.arange(len(S)), S.cluster.to_numpy()]
    for c in range(k):
        S[f"prob_c{c}"] = prob[:, c]  # renamed to prob_<archetype> after naming
    prof = S.groupby("cluster").apply(lambda g: pd.Series(np.average(g[FEATURES], axis=0, weights=g.minutes), index=FEATURES))
    names = name_clusters(prof)
    S["archetype"] = S.cluster.map(names)
    import pickle
    with open(PROC / "archetype_models.pkl", "wb") as fh:
        pickle.dump({"kmeans": km, "gmm": gm, "comp_of": comp_of, "names": names, "pos_weight": POS_WEIGHT}, fh)
    S = S.rename(columns={f"prob_c{c}": f"prob_{names[c]}" for c in range(k)})
    prof.index = prof.index.map(names)
    prof.round(3).to_csv(RESULTS / "step3_profiles.csv")
    S.to_parquet(PROC / "archetypes.parquet", index=False)
    share = S.groupby(["league", "archetype"]).minutes.sum() / S.groupby("league").minutes.sum()
    share.round(3).rename("minutes_share").to_csv(RESULTS / "step3_archetype_minutes_share.csv")
    labels = {"usage": "Usage", "ast": "Assist rate", "orb": "Off. rebound", "drb": "Def. rebound", "tov": "Turnovers",
              "stl": "Steals", "blk": "Blocks", "fta_rate": "FT trips", "assisted": "Assisted share",
              "rim_share": "Rim shots", "short_mid_share": "Short mid", "long_mid_share": "Long mid",
              "corner3_share": "Corner 3s"}
    order = ["On-ball creator", "Secondary guard", "Spot-up 3-and-D wing", "Connector forward",
             "Hub big / point forward", "Rim-running big"]
    P = prof.reindex([o for o in order if o in prof.index])
    fig, ax = plt.subplots(figsize=(11, 3.8), dpi=200)
    im = ax.imshow(P.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(FEATURES)), [labels[f] for f in FEATURES], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(P)), P.index, fontsize=9)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            v = P.iat[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 1.2 else "#0b0b0b")
    cb = fig.colorbar(im, ax=ax, fraction=.025, pad=.01)
    cb.set_label("SD vs league-season average", fontsize=8)
    ax.set_title("Six style archetypes (minutes-weighted average profile, both leagues)", fontsize=10, loc="left")
    plt.tight_layout()
    plt.savefig(FIGS / "step3_archetype_profiles.png", facecolor="#fcfcfb")
    print(names, file=sys.stderr)


def rebuild_style_matrix() -> None:
    """Fast path: style matrix + norms without re-running k selection."""
    d = pd.read_parquet(PROC / "player_season_shrunk.parquet")
    d = d[d.minutes > 0].reset_index(drop=True)
    norms = style_norms(d)
    norms.to_parquet(PROC / "style_norms.parquet")
    Z = style_matrix(d, norms)
    pd.concat([d[["league", "season", "athlete_id", "player", "position", "minutes"]], Z], axis=1).to_parquet(
        PROC / "style_matrix.parquet", index=False)


def fit_wnba_model(k: int = 6) -> None:
    """WNBA-only archetypes (the joint model is NBA-dominated: ARI 0.55 vs a WNBA-only fit).
    Names come from the joint model: each WNBA cluster takes the name of the joint-model
    centroid it is closest to (one-to-one matching), so both leagues share one vocabulary."""
    import pickle
    from scipy.optimize import linear_sum_assignment
    S = pd.read_parquet(PROC / "style_matrix.parquet")
    W = S[S.league == "WNBA"].reset_index(drop=True)
    X, w = model_matrix(W), fit_weight(W)
    km = KMeans(k, n_init=20, random_state=SEED).fit(X, sample_weight=w)
    with open(PROC / "archetype_models.pkl", "rb") as fh:
        M = pickle.load(fh)
    cost = ((km.cluster_centers_[:, None, :] - M["kmeans"].cluster_centers_[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(cost)
    names = {int(i): M["names"][int(j)] for i, j in zip(r, c)}
    W["archetype_wnba_model"] = pd.Series(km.predict(X)).map(names)
    W[["league", "season", "athlete_id", "player", "archetype_wnba_model"]].to_parquet(PROC / "archetypes_wnba_model.parquet", index=False)
    with open(PROC / "archetype_models_wnba.pkl", "wb") as fh:
        pickle.dump({"kmeans": km, "names": names}, fh)
    print("WNBA-only model names:", names, file=sys.stderr)
