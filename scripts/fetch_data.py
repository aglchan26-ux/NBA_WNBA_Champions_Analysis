"""Download ESPN game-level data (via sportsdataverse release files) for both leagues.

Usage: python scripts/fetch_data.py [--force]

Writes data/raw/espn/{league}_{kind}_{season}.parquet and data/raw/espn/MANIFEST.csv
(source URL, bytes, sha256, download time). Files that already exist are skipped
unless --force is given. A file only counts as downloaded if it starts with the
parquet magic bytes (missing release assets return a small "Not Found" body).
"""
import argparse
import hashlib
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "espn"
BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"

# ESPN "season" = calendar year the season ends (NBA 2026 = 2025-26).
SEASONS = {"nba": range(2004, 2027), "wnba": range(2009, 2027)}
KINDS = {
    "player_box": "espn_{lg}_player_boxscores/player_box_{y}.parquet",
    "team_box": "espn_{lg}_team_boxscores/team_box_{y}.parquet",
    "pbp": "espn_{lg}_pbp/play_by_play_{y}.parquet",
    "schedule": "espn_{lg}_schedules/{lg}_schedule_{y}.parquet",
}
ROSTER_SEASONS = {"nba": [2025, 2026, 2027], "wnba": [2024, 2025, 2026]}


def get(url: str, dest: Path, tries: int = 4) -> bool:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                data = r.read()
            if data[:4] != b"PAR1":
                return False
            dest.write_bytes(data)
            return True
        except Exception as e:  # network hiccup: back off and retry
            print(f"  retry {i + 1} {url}: {e}", file=sys.stderr)
            time.sleep(2 ** (i + 1))
    return False


EXTERNAL = {
    # FiveThirtyEight WNBA player stats 1997-2019 (from Basketball-Reference); used only for
    # player age (as of Jul 1). Repo has no LICENSE file; FiveThirtyEight data is published
    # under CC BY 4.0 in fivethirtyeight/data. Downloaded, not redistributed; attribute in use.
    "538_wnba_player_stats.csv":
        "https://raw.githubusercontent.com/fivethirtyeight/WNBA-stats/master/wnba-player-stats.csv",
    # NBA.com hustle stats (deflections, contests, charges drawn, loose balls, box-outs), NBA
    # 2017-18 onward, mirrored by the third-party repo gabriel1200/site_Data (no LICENSE file;
    # underlying data (c) NBA.com). Used for non-commercial analysis only; not redistributed.
    "nba_hustle.csv": "https://raw.githubusercontent.com/gabriel1200/site_Data/master/hustle.csv",
    "nba_hustle_playoffs.csv": "https://raw.githubusercontent.com/gabriel1200/site_Data/master/hustle_ps.csv",
    # Same third-party repo: NBA salaries 1991-2024 scraped from Spotrac / RealGM. Agrees with the
    # user's salary files within 5% on only ~67% of overlapping player-seasons, so it is used only
    # as a flagged fallback (salary_fallback) for player-seasons missing from the user's files.
    "nba_salaries_master.csv": "https://raw.githubusercontent.com/gabriel1200/site_Data/master/nba_salaries_master.csv",
}


def fetch_external(force: bool = False) -> None:
    ext = ROOT / "data" / "raw" / "external"
    ext.mkdir(parents=True, exist_ok=True)
    for name, url in EXTERNAL.items():
        dest = ext / name
        if dest.exists() and not force:
            continue
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(r.read())
        print(f"ok  external {name}", file=sys.stderr)


def main(force: bool = False) -> None:
    fetch_external(force)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for lg, years in SEASONS.items():
        for y in years:
            for kind, pat in KINDS.items():
                jobs.append((lg, kind, y, pat.format(lg=lg, y=y)))
        for y in ROSTER_SEASONS[lg]:
            jobs.append((lg, "roster", y, f"espn_{lg}_rosters/rosters_{y}.parquet"))
    rows = []
    for lg, kind, y, rel in jobs:
        dest = OUT / f"{lg}_{kind}_{y}.parquet"
        url = f"{BASE}/{rel}"
        ok = dest.exists() and not force
        if not ok:
            ok = get(url, dest)
            print(f"{'ok ' if ok else 'MISSING'} {lg} {kind} {y}", file=sys.stderr)
        rows.append({
            "league": lg, "kind": kind, "season": y, "url": url, "ok": ok,
            "bytes": dest.stat().st_size if ok else 0,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest() if ok else "",
            "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    man = pd.DataFrame(rows)
    man.to_csv(OUT / "MANIFEST.csv", index=False)
    print(man.groupby(["league", "kind"]).ok.sum().to_string(), file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
