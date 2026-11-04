"""Step 3e: validation summary for the archetype model (one table of estimates).

Adds the checks not computed elsewhere, then gathers every headline number into
results/step3_validation_summary.csv from saved result files:
  - stability of the final (position-aware) base model under minutes-weighted resampling
  - year-to-year consistency: same player, consecutive seasons, same team both years (1000+ min)
  - pooled vs per-league fits (ARI within each league)
  - drift: model fit on early seasons only, applied to later seasons, vs the full model
"""
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from common import PROC, RESULTS
from step3_archetypes import model_matrix, weighted_resample

SEED = 7
EARLY = {"NBA": 2014, "WNBA": 2017}  # last season of the "early" window (WNBA 2018 recording change)


def ci_prop(p, n):
    se = np.sqrt(p * (1 - p) / n)
    return max(0, p - 1.96 * se), min(1, p + 1.96 * se)


def main() -> None:
    S = pd.read_parquet(PROC / "style_matrix.parquet")
    A = pd.read_parquet(PROC / "archetypes_detailed.parquet")
    X, w = model_matrix(S), S.minutes.to_numpy()
    ref = KMeans(6, n_init=20, random_state=SEED).fit(X, sample_weight=w).predict(X)
    rng = np.random.default_rng(SEED)
    stab = [adjusted_rand_score(ref, KMeans(6, n_init=4, random_state=b).fit(X[weighted_resample(len(X), w, rng)]).predict(X))
            for b in range(10)]
    rows = [{"check": "Base archetype stability (bootstrap ARI, 10 resamples)", "estimate": np.mean(stab),
             "lo95": np.percentile(stab, 2.5), "hi95": np.percentile(stab, 97.5), "n": len(X),
             "how_to_read": "1 = identical clusters on resampled data; >0.9 is very stable"}]
    sub = pd.read_csv(RESULTS / "step3_subtypes.csv").drop_duplicates("archetype")
    kk = sub.apply(lambda r: r[f"ari_k{int(r.k_sub)}"] if r.k_sub > 1 else np.nan, axis=1)
    rows.append({"check": "Sub-type stability (bootstrap ARI, range over 6 archetypes)", "estimate": kk.median(),
                 "lo95": kk.min(), "hi95": kk.max(), "n": int(sub.k_sub.sum()),
                 "how_to_read": "lo/hi = weakest and strongest archetype split"})

    # year-to-year consistency
    ps = pd.read_parquet(PROC / "player_season.parquet", columns=["league", "season", "season_type", "athlete_id", "teams"])
    ps = ps[ps.season_type == "regular"]
    B = A.merge(ps, on=["league", "season", "athlete_id"])
    nxt = B.assign(season=B.season - 1)
    m = B.merge(nxt, on=["league", "season", "athlete_id"], suffixes=("", "_next"))
    m = m[(m.minutes >= 1000) & (m.minutes_next >= 1000) & (m.teams == m.teams_next) & ~m.teams.str.contains("\\|")]
    for lg, g in m.groupby("league"):
        p = (g.archetype == g.archetype_next).mean()
        lo, hi = ci_prop(p, len(g))
        rows.append({"check": f"{lg}: same archetype next season (same team, 1000+ min both years)", "estimate": p,
                     "lo95": lo, "hi95": hi, "n": len(g), "how_to_read": "roles persist but can evolve; ~0.6-0.8 is healthy"})

    # pooled vs per-league
    for lg in ["NBA", "WNBA"]:
        msk = (S.league == lg).to_numpy()
        own = KMeans(6, n_init=20, random_state=SEED).fit(X[msk], sample_weight=w[msk]).predict(X[msk])
        big = (S.minutes[msk] >= 500).to_numpy()
        a = adjusted_rand_score(ref[msk][big], own[big])
        rows.append({"check": f"{lg}: pooled model vs {lg}-only model (ARI, 500+ min)", "estimate": a, "lo95": np.nan,
                     "hi95": np.nan, "n": int(big.sum()), "how_to_read": "1 = the joint model sorts players exactly as a league-only model"})
    # drift: early-only model applied to later seasons
    early = np.array([s <= EARLY[lg] for lg, s in zip(S.league, S.season)])
    km_e = KMeans(6, n_init=20, random_state=SEED).fit(X[early], sample_weight=w[early])
    late = ~early & (S.minutes >= 500).to_numpy()
    a = adjusted_rand_score(ref[late], km_e.predict(X[late]))
    rows.append({"check": "Drift: early-seasons model applied to later seasons vs full model (ARI, 500+ min)",
                 "estimate": a, "lo95": np.nan, "hi95": np.nan, "n": int(late.sum()),
                 "how_to_read": "early = NBA <=2013-14, WNBA <=2017; high = archetype definitions stable over time"})
    sh_e = pd.Series(km_e.predict(X[~early]), index=S.index[~early]).value_counts(normalize=True).sort_index()
    sh_f = pd.Series(ref[~early]).value_counts(normalize=True).sort_index()

    # gather saved checks
    for batch, role in [("v1", "TUNING set"), ("v2", "UNTOUCHED check")]:
      f = RESULTS / f"step3_hand_label_agreement_{batch}.csv"
      if not f.exists():
          continue
      ha = pd.read_csv(f).set_index("group")
      for g, lab in [("all", "all"), ("league=NBA", "NBA"), ("league=WNBA", "WNBA"), ("definition=defined", "defined players"),
                   ("definition=blended", "blended players")]:
        if g in ha.index:
            p, n = ha.at[g, "archetype_agreement"], int(ha.at[g, "n"])
            lo, hi = ci_prop(p, n)
            rows.append({"check": f"Hand labels {batch} - archetype agreement, {lab} ({role})", "estimate": p,
                         "lo95": lo, "hi95": hi, "n": n,
                         "how_to_read": f"chance ~0.17; kappa {ha.at[g, 'cohen_kappa_archetype']:.2f}; top-2 {ha.at[g, 'hand_in_model_top2']:.2f}"})
    rc = pd.read_csv(RESULTS / "step3_role_change_rates.csv")
    for _, r in rc[rc.measure == "arch_changed"].iterrows():
        rows.append({"check": f"{r.league}: archetype changes, {r.comparison}", "estimate": r.rate, "lo95": r.lo95,
                     "hi95": r.hi95, "n": r.n_pairs, "how_to_read": "trades should change roles more often than the same-team baseline"})
    lg_ = pd.read_csv(RESULTS / "step3_trade_change_logit.csv").iloc[0]
    rows.append({"check": "NBA: odds ratio of archetype change, trade vs same team (adjusted for segment minutes)",
                 "estimate": lg_.odds_ratio_trade_vs_same_team, "lo95": lg_.lo95, "hi95": lg_.hi95, "n": int(lg_.n),
                 "how_to_read": ">1 means trades change roles beyond sampling noise"})
    hu = pd.read_csv(RESULTS / "step3_hustle_by_subtype.csv").set_index("subtype")
    for s, col, lab in [("Perimeter stopper forward", "deflections_z", "deflections"),
                        ("Defensive / connector guard", "deflections_z", "deflections"),
                        ("Low-usage rim protector", "contest2_z", "contested 2s"),
                        ("Defensive anchor big", "contest2_z", "contested 2s"),
                        ("Catch-and-shoot wing", "deflections_z", "deflections")]:
        if s in hu.index:
            rows.append({"check": f"NBA hustle check: {s} - {lab} (z vs league)", "estimate": hu.at[s, col], "lo95": np.nan,
                         "hi95": np.nan, "n": int(hu.at[s, "n"]),
                         "how_to_read": "independent NBA.com data (2017-18+); >0 supports a defensive label"})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "step3_validation_summary.csv", index=False)
    pd.DataFrame({"full_model": sh_f, "early_model": sh_e}).to_csv(RESULTS / "step3_drift_cluster_shares.csv")
    print(out[["check", "estimate", "lo95", "hi95", "n"]].round(3).to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
