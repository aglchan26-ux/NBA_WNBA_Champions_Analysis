"""Participation filter and injury-availability flags (regular season), one row per player-season.

Owner's rule (option C, adjusted):
  counted  = played >= 15% of team games
             OR (>= 20 min/game when playing AND >= MIN_GAMES games)   <- injury exemption
  MIN_GAMES: NBA 6, WNBA 4 (shorter season).
Rows that fail are NOT deleted: they stay in every table with counted = False, and are left out of
archetype fitting, named labels and value rankings.

Injury flags (inferred from playing time; ESPN box scores omit most inactive players, so the
'did not play' reason cannot be used to find injured players):
  injury_exempt          counted only because of the exemption (short season, real role)
  likely_season_ending   last game >= LATE_GAP team games before the team's final regular-season game,
                         >= 20 min/game, not a trade-deadline move (last team = team at season end)
  missed_window_games    team games missed between a player's first and last appearance
  availability           games played / team games (salary step: value per available game vs per season)
  absent_playoffs        team played in the playoffs, player had a counted regular season with
                         >= 20 min/game but 0 playoff minutes (title model: not counted on the playoff roster)
"""
import sys

import numpy as np
import pandas as pd

from common import PROC

SHARE = 0.15
EXEMPT_MPG = 20
MIN_GAMES = {"NBA": 6, "WNBA": 4}
LATE_GAP = {"NBA": 15, "WNBA": 6}


def build() -> pd.DataFrame:
    pg = pd.read_parquet(PROC / "player_game.parquet", columns=[
        "league", "season", "season_type", "game_id", "game_date", "athlete_id", "athlete_display_name",
        "team_id", "minutes"])
    reg = pg[pg.season_type == "regular"]
    team_dates = reg[["league", "season", "team_id", "game_id", "game_date"]].drop_duplicates()
    team_games = team_dates.groupby(["league", "season", "team_id"]).game_id.nunique()

    st = reg.groupby(["league", "season", "athlete_id", "team_id"]).agg(
        gp=("game_id", "nunique"), mins=("minutes", "sum"), first=("game_date", "min"),
        last=("game_date", "max"), player=("athlete_display_name", "first")).reset_index()
    j = st.merge(team_dates, on=["league", "season", "team_id"])
    j["in_window"] = j.game_date.between(j["first"], j["last"])
    j["after_last"] = j.game_date > j["last"]
    st = st.merge(j.groupby(["league", "season", "athlete_id", "team_id"]).agg(
        window=("in_window", "sum"), after=("after_last", "sum")).reset_index(),
        on=["league", "season", "athlete_id", "team_id"])
    st = st.sort_values("last")
    p = st.groupby(["league", "season", "athlete_id"]).agg(
        player=("player", "first"), gp=("gp", "sum"), minutes=("mins", "sum"), window=("window", "sum"),
        last_team=("team_id", "last"), games_after_last=("after", "last"), n_teams=("team_id", "nunique")).reset_index()
    p = p.merge(team_games.rename("last_team_games").reset_index().rename(columns={"team_id": "last_team"}),
                on=["league", "season", "last_team"])
    lg_games = team_games.groupby(["league", "season"]).median().rename("team_games")
    p = p.merge(lg_games.reset_index(), on=["league", "season"])

    p["share"] = p.gp / p.team_games
    p["mpg"] = p.minutes / p.gp
    p["availability"] = (p.gp / p.team_games).clip(upper=1)
    p["missed_window_games"] = p.window - p.gp
    min_games = p.league.map(MIN_GAMES)
    exempt = (p.mpg >= EXEMPT_MPG) & (p.gp >= min_games)
    p["counted"] = (p.share >= SHARE) | exempt
    p["injury_exempt"] = (p.share < SHARE) & exempt
    p["likely_season_ending"] = (p.games_after_last >= p.league.map(LATE_GAP)) & (p.mpg >= EXEMPT_MPG)

    po = pg[pg.season_type == "playoffs"]
    po_teams = po[["league", "season", "team_id"]].drop_duplicates().assign(team_in_playoffs=True)
    po_min = po.groupby(["league", "season", "athlete_id"]).minutes.sum().rename("playoff_minutes")
    p = p.merge(po_teams.rename(columns={"team_id": "last_team"}), on=["league", "season", "last_team"], how="left")
    p = p.merge(po_min.reset_index(), on=["league", "season", "athlete_id"], how="left")
    p["team_in_playoffs"] = p.team_in_playoffs.fillna(False).astype(bool)
    p["playoff_minutes"] = p.playoff_minutes.fillna(0)
    p["absent_playoffs"] = p.team_in_playoffs & p.counted & (p.mpg >= EXEMPT_MPG) & (p.playoff_minutes == 0)
    return p


def main() -> None:
    p = build()
    p.to_parquet(PROC / "participation.parquet", index=False)
    s = p.groupby("league").agg(player_seasons=("counted", "size"), not_counted=("counted", lambda x: (~x).sum()),
                                injury_exempt=("injury_exempt", "sum"),
                                likely_season_ending=("likely_season_ending", "sum"),
                                absent_playoffs=("absent_playoffs", "sum"))
    s["minutes_not_counted_pct"] = p.groupby("league").apply(
        lambda d: 100 * d.minutes[~d.counted].sum() / d.minutes.sum()).round(2)
    s.to_csv(PROC.parent.parent / "results" / "participation_summary.csv")
    print(s.to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()


def fit_weight(d: pd.DataFrame) -> np.ndarray:
    """Minutes as fitting weight, set to 0 for player-seasons that fail the participation rule.
    Those rows still get labels (predict), they just do not shape the clusters."""
    p = pd.read_parquet(PROC / "participation.parquet", columns=["league", "season", "athlete_id", "counted"])
    c = d[["league", "season", "athlete_id"]].merge(p, how="left", on=["league", "season", "athlete_id"]).counted
    return d.minutes.to_numpy() * c.fillna(True).astype(bool).to_numpy()
