"""Play-by-play extras per player-season (regular season and playoffs).

Shot creation (shooter = athlete_id_1 on shooting plays, free throws excluded), from ESPN shot labels:
  fga_drive     'Driving ...' shots
  fga_pullup    pull-up, step-back, fade-away jumpers
  fga_offball   cutting shots, putbacks, tips, alley-oops
  Labels are detailed only from NBA 2017-18 / WNBA 2019 on (before that ~70% of shots are a plain
  'Jump Shot'), so creation tags are produced only for those seasons (CREATION_FROM).
Defense (both leagues, all seasons):
  blk_rim / blk_jumper   blocks (blocker = athlete_id_2 on a blocked shot) of shots < 10 ft vs >= 10 ft
  stl_pass / stl_onball  steals on 'Bad Pass' turnovers vs 'Lost Ball' turnovers (stealer = athlete_id_2)
  shoot_fouls            shooting fouls committed (fouler = athlete_id_1)
"""
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import ESPN, PROC, SEASON_TYPE, SEASONS

CREATION_FROM = {"NBA": 2018, "WNBA": 2019}


def season_extras(lg: str, y: int, games: set) -> pd.DataFrame:
    f = ESPN / f"{lg}_pbp_{y}.parquet"
    cols = [c for c in ["game_id", "season_type", "type_text", "text", "shooting_play", "athlete_id_1", "athlete_id_2",
                        "coordinate_x", "coordinate_y"] if c in pq.read_schema(f).names]
    p = pd.read_parquet(f, columns=cols)
    p = p[p.game_id.isin(games) & p.season_type.isin(SEASON_TYPE)]
    tt, tx = p.type_text.fillna(""), p.text.fillna("")
    shot = (p.shooting_play == True) & ~tt.str.contains("Free Throw")  # noqa: E712
    x, yy = p.coordinate_x.astype(float).abs(), p.coordinate_y.astype(float)
    dist = np.sqrt((41.75 - x) ** 2 + yy ** 2)
    ev = []

    def add(mask, who, name):
        d = p.loc[mask, ["season_type", who]].rename(columns={who: "athlete_id"}).dropna()
        d["stat"] = name
        ev.append(d)

    add(shot & tt.str.contains("Driving"), "athlete_id_1", "fga_drive")
    add(shot & tt.str.contains("Pullup|Pull-up|Step Back|Fade Away|Fadeaway"), "athlete_id_1", "fga_pullup")
    add(shot & tt.str.contains("Cutting|Putback|Tip|Alley Oop"), "athlete_id_1", "fga_offball")
    blocked = shot & tx.str.contains(" blocks ")
    add(blocked & (dist < 10), "athlete_id_2", "blk_rim")
    add(blocked & (dist >= 10), "athlete_id_2", "blk_jumper")
    add(tt.str.contains("Bad Pass") & tx.str.contains("steals"), "athlete_id_2", "stl_pass")
    add(tt.str.contains("Lost Ball") & tx.str.contains("steals"), "athlete_id_2", "stl_onball")
    add(tt == "Shooting Foul", "athlete_id_1", "shoot_fouls")
    e = pd.concat(ev)
    out = e.groupby(["season_type", "athlete_id", "stat"]).size().unstack("stat", fill_value=0).reset_index()
    out["season_type"] = out.season_type.map(SEASON_TYPE)
    out["league"], out["season"] = lg.upper(), y
    return out


def main() -> None:
    pg = pd.read_parquet(PROC / "player_game.parquet", columns=["league", "season", "game_id"])
    parts = []
    for lg in SEASONS:
        for y in SEASONS[lg]:
            games = set(pg[(pg.league == lg.upper()) & (pg.season == y)].game_id)
            parts.append(season_extras(lg, y, games))
            print(f"{lg} {y}", file=sys.stderr, end=" ")
    out = pd.concat(parts, ignore_index=True).fillna(0)
    out["athlete_id"] = out.athlete_id.astype("int64")
    out.to_parquet(PROC / "player_pbp_extras.parquet", index=False)
    print(f"\nrows={len(out):,}", file=sys.stderr)


if __name__ == "__main__":
    main()
