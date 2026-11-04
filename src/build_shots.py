"""Step 1b: shot-level table from play-by-play, then player and team-defense zone counts.

Zones (distance from the nearest hoop, ESPN coordinates in feet, hoop at x = +/-41.75):
  rim        two-pointer, d < 4 ft (restricted-area size)
  short_mid  two-pointer, 4 <= d < 14
  long_mid   two-pointer, d >= 14
  corner3    three-pointer within 14 ft of the baseline (|x| >= 33) and |y| >= 20
  above3     other three-pointers
A three-pointer = play text says "three point" OR the shot is beyond the arc by geometry
(distance >= arc + 0.25 ft, or corner: |y| >= 22.25 ft and |x| >= 33). Arc = 23.75 ft NBA,
22.15 ft WNBA. Text alone missed 3-11% of threes (step-backs/pull-ups often omit it), and
the numeric points_attempted field disagrees with the text in NBA 2021-2024. The combined
rule matches box-score 3PA within about 1% (NBA) and 2% (WNBA) per season (see validation).
Assisted = made shot with a second athlete tagged on the play.
"""
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import ESPN, PROC, SEASONS, SEASON_TYPE

HOOP_X = 41.75
ZONES = ["rim", "short_mid", "long_mid", "corner3", "above3"]


def shots(lg: str, y: int) -> pd.DataFrame:
    f = ESPN / f"{lg}_pbp_{y}.parquet"
    cols = ["game_id", "season_type", "game_date", "type_text", "text", "shooting_play", "scoring_play",
            "team_id", "home_team_id", "away_team_id", "athlete_id_1", "athlete_id_2",
            "coordinate_x", "coordinate_y"]
    p = pd.read_parquet(f, columns=[c for c in cols if c in pq.read_schema(f).names])
    s = p[(p.shooting_play == True) & ~p.type_text.str.contains("Free Throw", na=False)  # noqa: E712
          & p.season_type.isin(SEASON_TYPE) & p.athlete_id_1.notna()].copy()
    x, yy = s.coordinate_x.astype(float), s.coordinate_y.astype(float)
    s["dist"] = np.sqrt((HOOP_X - x.abs()) ** 2 + yy ** 2)
    s["coord_ok"] = x.notna() & (x.abs() <= 47) & (yy.abs() <= 25)
    arc = (23.75 if lg == "nba" else 22.15) + 0.25
    geo3 = s.coord_ok & ((s.dist >= arc) | ((yy.abs() >= 22.25) & (x.abs() >= 33)))
    s["is3"] = s.text.str.contains("three point", case=False, na=False) | geo3
    s["made"] = s.scoring_play.fillna(False).astype(bool)
    s["assisted"] = s.made & s.athlete_id_2.notna()
    corner = (x.abs() >= 33) & (yy.abs() >= 20)
    s["zone"] = np.select(
        [s.is3 & corner, s.is3, s.dist < 4, s.dist < 14],
        ["corner3", "above3", "rim", "short_mid"], default="long_mid")
    s.loc[~s.coord_ok & ~s.is3, "zone"] = "unknown2"
    s["def_team_id"] = np.where(s.team_id == s.home_team_id, s.away_team_id, s.home_team_id)
    s["league"], s["season"] = lg.upper(), y
    s["season_type"] = s.season_type.map(SEASON_TYPE)
    return s[["league", "season", "season_type", "game_id", "game_date", "athlete_id_1", "athlete_id_2",
              "team_id", "def_team_id", "zone", "dist", "is3", "made", "assisted"]].rename(
        columns={"athlete_id_1": "athlete_id", "athlete_id_2": "assist_id"})


def zone_counts(s: pd.DataFrame, keys: list) -> pd.DataFrame:
    a = s.assign(n=1).pivot_table(index=keys, columns="zone", values="n", aggfunc="sum", fill_value=0)
    m = s.assign(n=s.made.astype(int)).pivot_table(index=keys, columns="zone", values="n", aggfunc="sum", fill_value=0)
    a.columns = [f"{c}_fga" for c in a.columns]
    m.columns = [f"{c}_fgm" for c in m.columns]
    out = a.join(m)
    out["pbp_fga"] = s.groupby(keys).size()
    out["pbp_fg3a"] = s[s.is3].groupby(keys).size()
    out["assisted_fgm"] = s[s.assisted].groupby(keys).size()
    out["pbp_fgm"] = s[s.made].groupby(keys).size()
    return out.fillna(0).reset_index()


def main() -> None:
    allshots = pd.concat([shots(lg, y) for lg in SEASONS for y in SEASONS[lg]], ignore_index=True)
    # Same exclusion as the box-score table (exhibitions, non-standings cup finals).
    games = pd.read_parquet(PROC / "player_game.parquet", columns=["game_id"]).game_id.unique()
    allshots = allshots[allshots.game_id.isin(games)]
    allshots.to_parquet(PROC / "shots.parquet", index=False)
    pz = zone_counts(allshots, ["league", "season", "season_type", "athlete_id"])
    pz.to_parquet(PROC / "player_zones.parquet", index=False)
    # Team defense: what opponents attempted and made against each team, by zone.
    dz = zone_counts(allshots.rename(columns={"team_id": "off_team_id"}),
                     ["league", "season", "season_type", "def_team_id"])
    dz.to_parquet(PROC / "team_defense_zones.parquet", index=False)
    oz = zone_counts(allshots, ["league", "season", "season_type", "team_id"])
    oz.to_parquet(PROC / "team_offense_zones.parquet", index=False)
    print(f"shots={len(allshots):,} player_zone_rows={len(pz):,}", file=sys.stderr)


if __name__ == "__main__":
    main()
