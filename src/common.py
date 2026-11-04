"""Shared constants and helpers."""
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ESPN = RAW / "espn"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

LEAGUES = ("nba", "wnba")
# NBA 2004 = 2003-04 (2002-03 play-by-play covers only ~81% of games).
# WNBA from 2009: earlier box scores have incomplete minutes (team totals do not add up).
SEASONS = {"nba": list(range(2004, 2027)), "wnba": list(range(2009, 2027))}
# Regulation minutes per team-game (5 players x game length).
REG_TEAM_MIN = {"nba": 240.0, "wnba": 200.0}
# Date used to compute age within a season: Feb 1 (NBA, season end year), Jul 1 (WNBA).
AGE_REF = {"nba": "{y}-02-01", "wnba": "{y}-07-01"}
SEASON_TYPE = {2: "regular", 3: "playoffs", 5: "play_in"}
# Bubble / shortened seasons flagged for robustness checks.
ATYPICAL = {("nba", 2012), ("nba", 2020), ("nba", 2021), ("wnba", 2020)}  # lockout, bubble, shortened

COUNT_COLS = [
    "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "orb", "drb", "ast", "stl", "blk", "tov", "pf", "pts",
]
ESPN_COUNT_MAP = {
    "field_goals_made": "fgm", "field_goals_attempted": "fga",
    "three_point_field_goals_made": "fg3m", "three_point_field_goals_attempted": "fg3a",
    "free_throws_made": "ftm", "free_throws_attempted": "fta",
    "offensive_rebounds": "orb", "defensive_rebounds": "drb", "assists": "ast",
    "steals": "stl", "blocks": "blk", "turnovers": "tov", "fouls": "pf", "points": "pts",
}


def norm_name(s) -> str:
    """Lowercase ASCII name key without suffixes or punctuation."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", " ", s)
    return re.sub(r"[^a-z]", "", s)
