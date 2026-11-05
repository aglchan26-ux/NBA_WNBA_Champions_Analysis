"""Export small, aggregated tables for the Quarto site (site/data/).

Only league/season/archetype-level summaries and model outputs are exported. No raw box scores,
salary sheets or player-level salary rows are copied, so the public site and repository do not
redistribute third-party data (Her Hoop Stats, Spotrac, Basketball-Reference, ESPN, NBA.com mirrors).
"""
import shutil
import sys

import numpy as np
import pandas as pd

from common import PROC, ROOT

OUT = ROOT / "site" / "data"
RES = ROOT / "results"
COPY = [
    "data_completeness_by_season.csv", "participation_summary.csv", "step1_salary_coverage.csv",
    "step2_validation.csv", "step3_validation_summary.csv", "step3_archetype_minutes_share.csv",
    "step3_subtypes.csv", "step5_champions_by_era.csv", "step5_playoff_zone_by_season.csv",
    "step5_playoff_zone_resilience.csv", "step5_B_title_cv.csv", "step5_A_netrating_cv.csv",
    "step5_A_ridge_coefs.csv", "step5_league_profiles.csv", "step5_wnba_years_behind.csv",
    "step5_champion_vs_league.csv", "step5_champion_identity.csv", "step5_title_no_net_cv.csv",
    "lineups_validation.csv", "player_impact_validation.csv", "step5_title_impact_cv.csv", "step5_series_cv.csv",
    "step5_series_significance.csv", "shot_distance_calibration.csv", "step6_cba_periods.csv", "player_impact_prior_comparison.csv", "step3_hand_label_agreement_v3.csv",
    "step5_final_title_cv.csv", "step5_final_series_cv.csv", "step5_final_significance.csv", "step5_final_weights.csv", "step4_topics.csv", "step4_text_vs_archetypes.csv",
    "step4_text_cluster_vs_archetype.csv", "step4_archetype_tokens.csv", "step6_counts.csv", "step6_model_cv.csv",
    "step6_top_players.csv", "step6_max_contracts.csv", "step6_lambda.csv", "step5_contenders_2027.csv", "step5_contenders_cv.csv", "step5_C_drift_windows.csv", "step5_checklist_by_group.csv",
]


def archetype_share_by_season() -> pd.DataFrame:
    a = pd.read_parquet(PROC / "archetypes_detailed.parquet", columns=["league", "season", "archetype", "minutes"])
    s = a.groupby(["league", "season", "archetype"]).minutes.sum()
    return (s / s.groupby(["league", "season"]).transform("sum")).rename("minutes_share").reset_index()


def top_impact(n: int = 10) -> pd.DataFrame:
    """Top players by lineup-based impact in each league's last two seasons (counted player-seasons only)."""
    P = pd.read_parquet(PROC / "player_impact.parquet")
    ps = pd.read_parquet(PROC / "player_season.parquet")
    ps = ps[ps.season_type == "regular"][["league", "season", "athlete_id", "player", "minutes"]]
    part = pd.read_parquet(PROC / "participation.parquet", columns=["league", "season", "athlete_id", "counted"])
    d = P.merge(ps, on=["league", "season", "athlete_id"]).merge(part, on=["league", "season", "athlete_id"])
    d = d[d.counted & (d.minutes >= 800)]
    last = d.groupby("league").season.transform("max")
    d = d[d.season >= last - 1]
    return (d.sort_values("impact", ascending=False).groupby(["league", "season"]).head(n)
            [["league", "season", "player", "o_impact", "d_impact", "impact", "minutes"]].round(1))


def champion_arch_share() -> pd.DataFrame:
    """Each champion's archetype minutes shares and the league-season average team share (for the mix chart)."""
    ts = pd.read_parquet(PROC / "team_seasons.parquet")
    arch = [c for c in ts.columns if c.startswith("arch_share::")]
    lg = ts.groupby(["league", "season"])[arch].mean().stack().rename("league_avg")
    ch = ts[ts.champion.astype(bool)].set_index(["league", "season"])[arch].stack().rename("champion")
    out = pd.concat([lg, ch], axis=1).reset_index()
    out.columns = ["league", "season", "archetype", "league_avg", "champion"]
    out["archetype"] = out.archetype.str.replace("arch_share::", "", regex=False)
    return out


def champion_position_share() -> pd.DataFrame:
    """Minutes share by listed position (G / F / C) for each champion vs the league-season team average."""
    pt = pd.read_parquet(PROC / "player_team_season.parquet")
    pt = pt[pt.season_type == "regular"][["league", "season", "athlete_id", "team_id", "minutes"]]
    pos = pd.read_parquet(PROC / "player_season.parquet")
    pos = pos[pos.season_type == "regular"][["league", "season", "athlete_id", "position"]]
    d = pt.merge(pos, on=["league", "season", "athlete_id"], how="left")
    d = d[d.position.isin(["G", "F", "C"])]
    sh = d.groupby(["league", "season", "team_id", "position"]).minutes.sum().unstack(fill_value=0)
    sh = sh.div(sh.sum(1), axis=0)
    ts = pd.read_parquet(PROC / "team_seasons.parquet")[["league", "season", "team_id", "champion"]]
    sh = sh.reset_index().merge(ts, on=["league", "season", "team_id"])
    lg = sh.groupby(["league", "season"])[["G", "F", "C"]].transform("mean")
    ch = sh[sh.champion.astype(bool)].copy()
    for p in ["G", "F", "C"]:
        ch[f"{p}_vs_league"] = ch[p] - lg.loc[ch.index, p]
    return ch[["league", "season", "G", "F", "C", "G_vs_league", "F_vs_league", "C_vs_league"]]


ROLE = {"Alpha scorer": "alpha engine", "Playmaking hub": "playmaking hub", "Floor general": "lead playmaker", "Shot-making scorer": "shot-maker",
        "Interior hub big": "post hub", "Stretch / skilled big": "skilled big", "Rim-running finisher": "rim finisher",
        "Low-usage rim protector": "rim protector", "Perimeter stopper forward": "perimeter stopper",
        "Scoring stretch forward": "scoring forward", "Microwave scoring guard": "scoring guard",
        "Defensive / connector guard": "two-way guard", "Catch-and-shoot wing": "shooter", "Movement shooter": "shooter",
        "Low-usage glue defender": "glue defender", "Rebounding / rim-running forward": "rebounding forward"}
DEPTH = {"Spot-up 3-and-D wing": "3-and-D wings", "Rim-running big": "rim-running bigs", "Connector forward": "connector forwards",
         "Secondary guard": "secondary guards", "On-ball creator": "extra creators", "Hub big / point forward": "hub bigs"}


def champion_profiles() -> pd.DataFrame:
    """One line per champion: identity, best player (lineup impact) and role, and the role it leaned on most."""
    ci = pd.read_csv(RES / "step5_champion_identity.csv")
    P = pd.read_parquet(PROC / "player_impact.parquet")[["league", "season", "athlete_id", "impact"]]
    pt = pd.read_parquet(PROC / "player_team_season.parquet")
    pt = pt[pt.season_type == "regular"][["league", "season", "athlete_id", "team_id", "minutes"]]
    A = pd.read_parquet(PROC / "archetypes_detailed.parquet", columns=["league", "season", "athlete_id", "player", "subtype", "archetype"])
    ts = pd.read_parquet(PROC / "team_seasons.parquet")[["league", "season", "team_id", "team", "champion"]]
    ch = ts[ts.champion.astype(bool)]
    d = pt.merge(ch[["league", "season", "team_id", "team"]]).merge(P).merge(A)
    d = d[d.minutes >= d.league.map({"NBA": 1000, "WNBA": 500})]           # WNBA seasons are 34-44 games
    # league rank of each player's impact that season (same minutes floor), to tell true star engines from committees
    allp = pt.merge(P).groupby(["league", "season", "athlete_id"]).agg(minutes=("minutes", "sum"), impact=("impact", "first")).reset_index()
    allp = allp[allp.minutes >= allp.league.map({"NBA": 1000, "WNBA": 500})]
    allp["league_rank"] = allp.groupby(["league", "season"]).impact.rank(ascending=False)
    d = d.merge(allp[["league", "season", "athlete_id", "league_rank"]], on=["league", "season", "athlete_id"], how="left")
    d = d.sort_values("impact", ascending=False).groupby(["league", "season"]).head(1)
    # an alpha scorer who plays as a hub big (e.g. Jokic) is a playmaking hub, not a scoring guard-type engine
    d.loc[(d.subtype == "Alpha scorer") & (d.archetype == "Hub big / point forward"), "subtype"] = "Playmaking hub"
    out = ci.merge(d[["league", "season", "player", "subtype", "impact", "league_rank"]], on=["league", "season"], how="left")
    # role emphasis in standard deviations: (champion share - league mean) / spread across teams that season, so a role
    # every team plays a lot (e.g. 3-and-D wings) can't dominate just because its raw percentage-point gaps are bigger
    ts_all = pd.read_parquet(PROC / "team_seasons.parquet")
    archs = [c for c in ts_all.columns if c.startswith("arch_share::")]
    g = ts_all.groupby(["league", "season"])[archs]
    z = (ts_all[archs] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    z.columns = [c.replace("arch_share::", "z::") for c in archs]
    zc = pd.concat([ts_all[["league", "season", "champion"]], z], axis=1)
    zc = zc[zc.champion.astype(bool)].drop(columns="champion")
    out = out.merge(zc, on=["league", "season"], how="left")
    arch = [c for c in ci.columns if c.startswith("arch_vs_league::")]
    rows = []
    for r in out.itertuples():
        lead = r.identity.capitalize()
        if r.identity == "balanced" and r.off_edge >= 4 and r.def_edge >= 4:
            lead = "Two-way"
        star = f"{str(r.player).split(' ', 1)[-1]} {ROLE.get(r.subtype, str(r.subtype).lower())}" if isinstance(r.player, str) else "deep roster"
        top = max(arch, key=lambda c: out.loc[r.Index, c])
        rows.append({"league": r.league, "season": r.season, "team": r.team, "identity": r.identity,
                     "role": ROLE.get(r.subtype, str(r.subtype).lower()) if isinstance(r.player, str) else "deep roster",
                     "off_edge": r.off_edge, "def_edge": r.def_edge,
                     "best_player": r.player, "best_player_impact": r.impact, "best_player_rank": r.league_rank,
                     "lean": top.split("::")[1],
                     **{c: out.loc[r.Index, c] for c in out.columns if c.startswith("z::")},
                     "summary": f"{lead}, {star}, deep {DEPTH[top.split('::')[1]]}"})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in COPY:
        shutil.copy(RES / f, OUT / f)
    (ROOT / "site" / "figures").mkdir(exist_ok=True)
    shutil.copy(ROOT / "figures" / "step3_archetype_profiles.png", ROOT / "site" / "figures")
    champion_position_share().to_csv(OUT / "champion_position_share.csv", index=False)
    champion_profiles().to_csv(OUT / "champion_profiles.csv", index=False)
    champion_arch_share().to_csv(OUT / "champion_arch_share.csv", index=False)
    top_impact().to_csv(OUT / "top_player_impact.csv", index=False)
    archetype_share_by_season().to_csv(OUT / "archetype_share_by_season.csv", index=False)
    print(f"exported {len(COPY) + 1} tables to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
