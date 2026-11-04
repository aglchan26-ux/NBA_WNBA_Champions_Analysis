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
    "step5_A_ridge_coefs.csv", "step5_C_drift_windows.csv", "step5_checklist_by_group.csv",
]


def archetype_share_by_season() -> pd.DataFrame:
    a = pd.read_parquet(PROC / "archetypes_detailed.parquet", columns=["league", "season", "archetype", "minutes"])
    s = a.groupby(["league", "season", "archetype"]).minutes.sum()
    return (s / s.groupby(["league", "season"]).transform("sum")).rename("minutes_share").reset_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in COPY:
        shutil.copy(RES / f, OUT / f)
    (ROOT / "site" / "figures").mkdir(exist_ok=True)
    shutil.copy(ROOT / "figures" / "step3_archetype_profiles.png", ROOT / "site" / "figures")
    archetype_share_by_season().to_csv(OUT / "archetype_share_by_season.csv", index=False)
    print(f"exported {len(COPY) + 1} tables to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
