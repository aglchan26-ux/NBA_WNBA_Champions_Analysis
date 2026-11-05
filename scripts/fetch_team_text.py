"""Fetch team-season articles from Wikipedia (plain text) for the reputation / sentiment layer.

Run this on your own computer (Wikipedia is not reachable from the cloud build environment):
    python scripts/fetch_team_text.py                 # champions and finalists (about 80 pages)
    python scripts/fetch_team_text.py --all-playoff   # every playoff team (about 600 pages)
    python scripts/fetch_team_text.py --dry-run       # list the article titles only

Uses the official MediaWiki API (action=query, prop=extracts, plain text), one request per second, with a
descriptive User-Agent, as Wikipedia's API etiquette asks. Article text is CC BY-SA; it is cached locally in
data/raw/text/wikipedia/ and NOT committed or republished (only derived scores are).
Title pattern: NBA "2024–25 Boston Celtics season" (en dash), WNBA "2025 Las Vegas Aces season".
Pages that do not exist are recorded as missing; nothing is guessed.
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "text" / "wikipedia"
API = "https://en.wikipedia.org/w/api.php"
UA = "hoops-value-research/0.1 (non-commercial sports analytics; https://github.com/aglchan26-ux)"


def team_names() -> pd.DataFrame:
    rows = []
    for f in sorted((ROOT / "data" / "raw" / "espn").glob("*_schedule_*.parquet")):
        lg, season = f.name.split("_")[0].upper(), int(f.stem.split("_")[-1])
        s = pd.read_parquet(f, columns=["home_id", "home_display_name"])
        s = s.drop_duplicates("home_id")
        rows.append(pd.DataFrame({"league": lg, "season": season, "team_id": s.home_id.astype(int), "name": s.home_display_name}))
    return pd.concat(rows)


def title(league: str, season: int, name: str) -> str:
    if league == "NBA":
        return f"{season - 1}–{str(season)[-2:]} {name} season"
    return f"{season} {name} season"


def targets(all_playoff: bool) -> pd.DataFrame:
    ts = pd.read_parquet(ROOT / "data" / "processed" / "team_seasons.parquet",
                         columns=["league", "season", "team_id", "champion", "finalist", "made_playoffs"])
    keep = ts.made_playoffs if all_playoff else (ts.champion.astype(bool) | ts.finalist.astype(bool))
    t = ts[keep].merge(team_names(), on=["league", "season", "team_id"], how="left").dropna(subset=["name"])
    t["title"] = [title(lg, s, n) for lg, s, n in zip(t.league, t.season, t.name)]
    return t


def fetch(title_: str) -> dict:
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                                "format": "json", "formatversion": 2, "titles": title_})
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        page = json.load(r)["query"]["pages"][0]
    return {"title": title_, "missing": bool(page.get("missing")), "text": page.get("extract", "")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-playoff", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    t = targets(a.all_playoff)
    if a.dry_run:
        print(t[["league", "season", "title"]].to_string(index=False))
        print(f"{len(t)} titles", file=sys.stderr)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for r in t.itertuples():
        dest = OUT / f"{r.league}_{r.season}_{r.team_id}.json"
        if dest.exists():
            continue
        for attempt in range(4):
            try:
                d = fetch(r.title)
                break
            except Exception as e:  # rate limit or network hiccup: back off
                print(f"retry {attempt + 1} {r.title}: {e}", file=sys.stderr)
                time.sleep(2 ** (attempt + 2))
        else:
            continue
        d.update({"league": r.league, "season": int(r.season), "team_id": int(r.team_id)})
        dest.write_text(json.dumps(d), encoding="utf-8")
        print(f"{'missing' if d['missing'] else 'ok':7s} {r.title}", file=sys.stderr)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
