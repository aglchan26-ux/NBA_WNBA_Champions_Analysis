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
    "step5_series_significance.csv", "shot_distance_calibration.csv", "step4_topics.csv", "step4_text_vs_archetypes.csv",
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in COPY:
        shutil.copy(RES / f, OUT / f)
    (ROOT / "site" / "figures").mkdir(exist_ok=True)
    shutil.copy(ROOT / "figures" / "step3_archetype_profiles.png", ROOT / "site" / "figures")
    top_impact().to_csv(OUT / "top_player_impact.csv", index=False)
    archetype_share_by_season().to_csv(OUT / "archetype_share_by_season.csv", index=False)
    print(f"exported {len(COPY) + 1} tables to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
