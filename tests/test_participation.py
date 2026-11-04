import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import PROC  # noqa: E402


def test_participation_rule():
    p = pd.read_parquet(PROC / "participation.parquet")
    mg = p.league.map({"NBA": 6, "WNBA": 4})
    expect = (p.gp / p.team_games >= 0.15) | ((p.minutes / p.gp >= 20) & (p.gp >= mg))
    assert (p.counted == expect).all()
    assert not p[p.injury_exempt].eval("gp / team_games >= 0.15").any()
    # one row per player-season, nothing deleted
    assert not p.duplicated(["league", "season", "athlete_id"]).any()
    ps = pd.read_parquet(PROC / "player_season.parquet")
    assert len(p) == (ps.season_type == "regular").sum()
