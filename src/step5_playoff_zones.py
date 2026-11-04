"""Do some shot types hold up better in the playoffs? (association, not causation)

For every team that made the playoffs, compare its own regular-season points per shot by zone
with its playoff points per shot by zone (same team, same season), plus how its shot mix shifts.
Zones: rim (< 4 ft), paint/short mid (4-14 ft), long mid (14 ft to arc), three. Drives use the
ESPN 'Driving' shot label (NBA 2017-18+, WNBA 2019+). Intervals: bootstrap over seasons.
"""
import sys

import numpy as np
import pandas as pd

from common import PROC, RESULTS

ZMAP = {"rim": "rim", "short_mid": "paint / short mid", "long_mid": "long mid", "corner3": "three", "above3": "three"}
VAL = {"rim": 2, "paint / short mid": 2, "long mid": 2, "three": 3}


def main() -> None:
    s = pd.read_parquet(PROC / "shots.parquet", columns=["league", "season", "season_type", "team_id", "zone", "made"])
    s = s[s.zone.isin(ZMAP) & s.season_type.isin(["regular", "playoffs"])]
    s["z"] = s.zone.map(ZMAP)
    po_teams = s[s.season_type == "playoffs"][["league", "season", "team_id"]].drop_duplicates()
    s = s.merge(po_teams, on=["league", "season", "team_id"])
    g = s.groupby(["league", "season", "season_type", "z"]).made.agg(["sum", "size"]).reset_index()
    g["pps"] = g["sum"] / g["size"] * g.z.map(VAL)
    tot = g.groupby(["league", "season", "season_type"])["size"].transform("sum")
    g["share"] = g["size"] / tot
    w = g.pivot_table(index=["league", "season", "z"], columns="season_type", values=["pps", "share"]).reset_index()
    w.columns = ["league", "season", "zone", "pps_playoffs", "pps_regular", "share_playoffs", "share_regular"]
    w["pps_change"] = w.pps_playoffs - w.pps_regular
    w["share_change"] = w.share_playoffs - w.share_regular
    w.to_csv(RESULTS / "step5_playoff_zone_by_season.csv", index=False)
    rng = np.random.default_rng(5)
    rows = []
    for (lg, z), h in w.groupby(["league", "zone"]):
        bs = [h.sample(len(h), replace=True, random_state=int(rng.integers(1e9))).pps_change.mean() for _ in range(2000)]
        rows.append({"league": lg, "zone": z, "seasons": len(h), "pps_regular": h.pps_regular.mean(),
                     "pps_playoffs": h.pps_playoffs.mean(), "pps_change": h.pps_change.mean(),
                     "lo95": np.percentile(bs, 2.5), "hi95": np.percentile(bs, 97.5),
                     "share_change_pts": 100 * h.share_change.mean(),
                     "seasons_worse_in_playoffs": (h.pps_change < 0).mean()})
    out = pd.DataFrame(rows)
    # drives (modern seasons only)
    x = pd.read_parquet(PROC / "player_pbp_extras.parquet")
    ps = pd.read_parquet(PROC / "player_season.parquet", columns=["league", "season", "season_type", "athlete_id", "fga"])
    m = x.merge(ps, on=["league", "season", "season_type", "athlete_id"])
    m = m[((m.league == "NBA") & (m.season >= 2018)) | ((m.league == "WNBA") & (m.season >= 2019))]
    d = m.groupby(["league", "season", "season_type"])[["fga_drive", "fga"]].sum()
    d["drive_share"] = d.fga_drive / d.fga
    dd = d.drive_share.unstack("season_type")
    for lg, h in dd.groupby(level=0):
        out = pd.concat([out, pd.DataFrame([{"league": lg, "zone": "drives (share of FGA)", "seasons": len(h),
                                             "share_change_pts": 100 * (h.playoffs - h.regular).mean()}])])
    out.to_csv(RESULTS / "step5_playoff_zone_resilience.csv", index=False)
    print(out.round(3).to_string(index=False), file=sys.stderr)


def team_profile_effect() -> None:
    """Do three-heavy or rim-heavy teams lose more offense from regular season to playoffs?
    Team-level: change in points per 100 possessions (playoffs minus regular season) regressed on
    the team's regular-season three-rate and rim-rate (z within league-season), season fixed
    effects, errors clustered by season. Also FTA/FGA and TOV/FGA changes (shots ignore FTs)."""
    import statsmodels.formula.api as smf
    pg = pd.read_parquet(PROC / "player_game.parquet", columns=["league", "season", "season_type", "team_id", "game_id",
                                                                "poss", "t_pts", "fga", "fta", "tov"])
    po = pg[pg.season_type == "playoffs"][["league", "season", "team_id"]].drop_duplicates()
    x = pg[pg.season_type.isin(["regular", "playoffs"])].merge(po)
    t = x.groupby(["league", "season", "season_type"])[["fga", "fta", "tov"]].sum()
    t["fta_per_fga"], t["tov_per_fga"] = t.fta / t.fga, t.tov / t.fga
    tt = t[["fta_per_fga", "tov_per_fga"]].unstack("season_type")
    ft_rows = []
    for lg in tt.index.get_level_values(0).unique():
        h = tt.loc[lg]
        ch = h.xs("playoffs", axis=1, level=1) - h.xs("regular", axis=1, level=1)
        ft_rows.append({"league": lg, "fta_per_fga_change": ch.fta_per_fga.mean(), "tov_per_fga_change": ch.tov_per_fga.mean(),
                        "seasons_more_fta_in_playoffs": (ch.fta_per_fga > 0).mean()})
    pd.DataFrame(ft_rows).to_csv(RESULTS / "step5_playoff_ft_tov_change.csv", index=False)
    tg = x.drop_duplicates(["game_id", "team_id"])
    e = tg.groupby(["league", "season", "team_id", "season_type"]).apply(lambda g: 100 * g.t_pts.sum() / g.poss.sum())
    e = e.unstack("season_type").dropna()
    e["ortg_change"] = e.playoffs - e.regular
    z = pd.read_parquet(PROC / "team_offense_zones.parquet")
    z = z[z.season_type == "regular"].copy()
    z["rim_rate"] = z.rim_fga / z.pbp_fga
    z["three_rate"] = (z.corner3_fga + z.above3_fga) / z.pbp_fga
    m = e.reset_index().merge(z[["league", "season", "team_id", "rim_rate", "three_rate"]], on=["league", "season", "team_id"])
    for c in ["rim_rate", "three_rate"]:
        m[c + "_z"] = m.groupby(["league", "season"])[c].transform(lambda v: (v - v.mean()) / v.std())
    m = m.dropna(subset=["ortg_change", "rim_rate_z", "three_rate_z"]).reset_index(drop=True)
    m.to_csv(RESULTS / "step5_team_playoff_ortg_change.csv", index=False)
    rows = []
    for lg, g in m.groupby("league"):
        g = g.reset_index(drop=True)
        f = smf.ols("ortg_change ~ three_rate_z + rim_rate_z + C(season)", data=g).fit(
            cov_type="cluster", cov_kwds={"groups": g.season.astype("category").cat.codes.to_numpy()})
        for v in ["three_rate_z", "rim_rate_z"]:
            lo, hi = f.conf_int().loc[v]
            rows.append({"league": lg, "term": v, "ortg_change_per_sd": f.params[v], "lo95": lo, "hi95": hi,
                         "n_teams": len(g), "mean_ortg_change": g.ortg_change.mean()})
    r = pd.DataFrame(rows)
    r.to_csv(RESULTS / "step5_shot_profile_playoff_ortg.csv", index=False)
    print(pd.DataFrame(ft_rows).round(3).to_string(index=False), file=sys.stderr)
    print(r.round(2).to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
    team_profile_effect()
