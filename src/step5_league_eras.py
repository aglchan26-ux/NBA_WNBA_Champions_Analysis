"""Which NBA era is the WNBA in now? League and champion shot profiles, compared across leagues.

1. League-season profile (regular season, all teams pooled): share of field-goal attempts at the rim,
   short mid, long mid, corner three and above-the-break three, plus share of made shots assisted.
   Shares of attempts are comparable across leagues even though the WNBA three-point line is shorter
   (22.15 ft vs 23.75 ft); the comparison is "how a league chooses its shots", not distance.
2. Era match: each WNBA season is matched to the closest NBA season (Euclidean distance on the profile,
   each feature standardized on all league-seasons), giving an "NBA-equivalent season" and its lag.
   A bootstrap over WNBA teams gives a range for the matched season.
3. Champion identity: the champion's profile minus its own league's profile (how champions differ from
   the league they won), and the champion's archetype mix minus the league's.
Outputs: results/step5_league_profiles.csv, step5_wnba_nba_era_match.csv, step5_champion_vs_league.csv
"""
import sys

import numpy as np
import pandas as pd

from common import PROC, RESULTS

ZONES = ["rim", "short_mid", "long_mid", "corner3", "above3"]
FEATS = [f"{z}_share" for z in ZONES] + ["assisted_share"]
SEED = 7


def profile(t: pd.DataFrame) -> pd.Series:
    fga = t[[f"{z}_fga" for z in ZONES]].sum()
    out = {f"{z}_share": fga[f"{z}_fga"] / fga.sum() for z in ZONES}
    out["assisted_share"] = t.assisted_fgm.sum() / t.pbp_fgm.sum()
    out["three_share"] = out["corner3_share"] + out["above3_share"]
    out["mid_share"] = out["short_mid_share"] + out["long_mid_share"]
    return pd.Series(out)


def lag_by_feature(lg: pd.DataFrame) -> pd.DataFrame:
    """For each WNBA season and each shot-share feature, the (interpolated) NBA season with the same value.
    Only meaningful where the NBA series moved steadily (threes up, mid-range down); other features
    are reported but flagged when the WNBA value lies outside the NBA range."""
    nba = lg[lg.league == "NBA"].set_index("season").sort_index()
    rows = []
    for _, r in lg[lg.league == "WNBA"].iterrows():
        for f in ["three_share", "mid_share", "rim_share", "assisted_share"]:
            x = nba[f].rolling(3, center=True, min_periods=1).mean()  # smooth single-season noise
            inc = x.iloc[-1] > x.iloc[0]
            xs = x if inc else -x
            v = r[f] if inc else -r[f]
            mono = xs.cummax()
            if v < mono.iloc[0] or v > mono.iloc[-1]:
                eq = np.nan
            else:
                eq = float(np.interp(v, mono.to_numpy(), mono.index.to_numpy().astype(float)))
            rows.append({"wnba_season": int(r.season), "feature": f, "wnba_value": r[f],
                         "nba_equivalent_season": eq, "years_behind": (r.season - eq) if eq == eq else np.nan})
    return pd.DataFrame(rows)


def champion_identity(ts: pd.DataFrame) -> pd.DataFrame:
    """Where each champion sat on offense and defense relative to its own league that regular season.
    identity: offense-led / defense-led when one side's edge exceeds the other's by >= 3 points per 100
    possessions; otherwise balanced. Ranks are within the league-season (1 = best)."""
    ts = ts.copy()
    g = ts.groupby(["league", "season"])
    ts["off_edge"] = ts.ortg - g.ortg.transform("mean")          # + = better offense
    ts["def_edge"] = g.drtg.transform("mean") - ts.drtg          # + = better defense
    ts["off_rank"] = g.ortg.rank(ascending=False, method="min")
    ts["def_rank"] = g.drtg.rank(ascending=True, method="min")
    ts["n_teams"] = g.team_id.transform("count")
    c = ts[ts.champion == 1].copy()
    gap = c.off_edge - c.def_edge
    c["identity"] = np.select([gap >= 3, gap <= -3], ["offense-led", "defense-led"], "balanced")
    arch = [x for x in ts.columns if x.startswith("arch_share::")]
    lg_arch = g[arch].transform("mean")
    for a in arch:
        c[a.replace("arch_share::", "arch_vs_league::")] = (ts[a] - lg_arch[a]).loc[c.index]
    keep = ["league", "season", "team", "off_edge", "def_edge", "off_rank", "def_rank", "n_teams", "identity"]
    return c[keep + [x for x in c.columns if x.startswith("arch_vs_league::")]]


def main() -> None:
    t = pd.read_parquet(PROC / "team_offense_zones.parquet")
    t = t[t.season_type == "regular"]
    ts = pd.read_parquet(PROC / "team_seasons.parquet")
    lg = t.groupby(["league", "season"]).apply(profile).reset_index()
    lg.to_csv(RESULTS / "step5_league_profiles.csv", index=False)
    lag_by_feature(lg).to_csv(RESULTS / "step5_wnba_years_behind.csv", index=False)

    mu, sd = lg[FEATS].mean(), lg[FEATS].std()
    nba = lg[lg.league == "NBA"].reset_index(drop=True)
    Zn = ((nba[FEATS] - mu) / sd).to_numpy()
    rng = np.random.default_rng(SEED)
    rows = []
    for s, g in t[t.league == "WNBA"].groupby("season"):
        z = ((profile(g)[FEATS] - mu) / sd).to_numpy()
        d = np.sqrt(((Zn - z) ** 2).sum(1))
        best = int(nba.season.iat[d.argmin()])
        boots = []
        teams = g.team_id.unique()
        for _ in range(500):
            gb = pd.concat([g[g.team_id == x] for x in rng.choice(teams, len(teams))])
            zb = ((profile(gb)[FEATS] - mu) / sd).to_numpy()
            boots.append(int(nba.season.iat[np.sqrt(((Zn - zb) ** 2).sum(1)).argmin()]))
        rows.append({"wnba_season": s, "nba_equivalent": best, "lag_years": s - best,
                     "nba_equiv_lo90": int(np.percentile(boots, 5)), "nba_equiv_hi90": int(np.percentile(boots, 95)),
                     "distance": round(float(d.min()), 2)})
    em = pd.DataFrame(rows)
    champion_identity(ts).to_csv(RESULTS / "step5_champion_identity.csv", index=False)
    em.to_csv(RESULTS / "step5_wnba_nba_era_match.csv", index=False)

    champs = ts[ts.champion == 1][["league", "season", "team_id", "team"]]
    tp = t.groupby(["league", "season", "team_id"]).apply(profile).reset_index()
    c = champs.merge(tp, on=["league", "season", "team_id"]).merge(lg, on=["league", "season"], suffixes=("", "_lg"))
    for f in FEATS + ["three_share", "mid_share"]:
        c[f"{f}_vs_league"] = c[f] - c[f"{f}_lg"]
    arch = [x for x in ts.columns if x.startswith("arch_share::")]
    lg_arch = ts.groupby(["league", "season"])[arch].mean()
    ca = ts[ts.champion == 1].set_index(["league", "season"])[arch] - lg_arch.loc[ts[ts.champion == 1].set_index(["league", "season"]).index]
    ca.columns = [x.replace("arch_share::", "arch_vs_league::") for x in ca.columns]
    c = c.merge(ca.reset_index(), on=["league", "season"], how="left")
    keep = ["league", "season", "team"] + [f"{f}_vs_league" for f in FEATS + ["three_share", "mid_share"]] + list(ca.columns)
    c[keep].to_csv(RESULTS / "step5_champion_vs_league.csv", index=False)
    print(em.to_string(index=False), file=sys.stderr)
    print(c[["league", "season", "team", "three_share_vs_league", "rim_share_vs_league", "mid_share_vs_league"]].round(3).tail(12).to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
