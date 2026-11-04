"""Step 1a: player-game -> player-team-season and player-season tables (both leagues).

Exposure columns (the denominators used in Step 2):
  poss_on        team possessions while the player was on the floor
                 = team-game possessions x (player minutes / (team minutes / 5))
  orb_chances_on available offensive rebounds on floor  = (team ORB + opp DRB) x share
  drb_chances_on available defensive rebounds on floor  = (team DRB + opp ORB) x share
  tm_fgm_on      teammate field goals made on floor     = team FGM x share - own FGM
  opp_fga_on     opponent FGA on floor (block exposure)  = opp FGA x share
Team-game possessions use the standard estimate averaged over both teams:
  poss = 0.5 * [(FGA + 0.44 FTA - ORB + TOV)_team + (same)_opp]
with team totals summed from player rows (see team_games for why).
"""
import sys

import numpy as np
import pandas as pd

from common import (ESPN, ESPN_COUNT_MAP, COUNT_COLS, PROC, SEASONS, SEASON_TYPE,
                    ATYPICAL, AGE_REF, RAW, norm_name)


def team_only_turnovers(lg: str, y: int, player_tov: pd.Series) -> pd.Series:
    """Team turnovers not credited to a player (shot clock, 8-second, etc.) per team-game.
    ESPN team files label these inconsistently across seasons ('turnovers' sometimes already
    includes them, 'total_turnovers' is sometimes doubled or missing). Rule: for each candidate
    team-file field, team-only = field - sum of player turnovers; keep the largest value in
    [0, 5]. If none is plausible, use the league-season median of accepted values."""
    t = pd.read_parquet(ESPN / f"{lg}_team_box_{y}.parquet")
    t = t.drop_duplicates(["game_id", "team_id"]).set_index(["game_id", "team_id"])
    ps = player_tov.reindex(t.index)
    cands = []
    for c in ("turnovers", "total_turnovers"):
        if c in t:
            d = pd.to_numeric(t[c], errors="coerce") - ps
            cands.append(d.where(d.between(0, 5)))
    if "team_turnovers" in t:
        tt = pd.to_numeric(t["team_turnovers"], errors="coerce")
        cands.append(tt.where(tt.between(0, 5)))
    val = pd.concat(cands, axis=1).max(axis=1) if cands else pd.Series(np.nan, index=t.index)
    # No plausible value in the whole file (NBA 2017-18): median of all other seasons,
    # measured with this same rule (NBA 0.65, WNBA 0.90 team-only turnovers per game).
    fallback = val.median() if val.notna().any() else {"nba": 0.65, "wnba": 0.90}[lg]
    return val.fillna(fallback)


def team_games(b: pd.DataFrame, lg: str, y: int) -> pd.DataFrame:
    """Team-game totals summed from player rows (consistent in every season; older ESPN
    team files have wrong ORB and turnover totals), plus team-only turnovers."""
    sums = {"fgm": "t_fgm", "fga": "t_fga", "fta": "t_fta", "orb": "t_orb", "drb": "t_drb", "tov": "t_tov", "pts": "t_pts"}
    t = b.groupby(["game_id", "team_id"])[list(sums)].sum().rename(columns=sums)
    tto = team_only_turnovers(lg, y, t.t_tov)
    t["t_team_tov"] = tto.reindex(t.index).fillna(tto.median() if len(tto) else 0.0).to_numpy()
    t["t_tov"] = t.t_tov + t.t_team_tov
    t = t.reset_index()
    opp = t[["game_id", "team_id", *sums.values()]].rename(
        columns={"team_id": "opponent_team_id", **{v: v.replace("t_", "o_") for v in sums.values()}})
    t = t.merge(opp, on="game_id")
    t = t[t.team_id != t.opponent_team_id]
    own = t.t_fga + 0.44 * t.t_fta - t.t_orb + t.t_tov
    other = t.o_fga + 0.44 * t.o_fta - t.o_orb + t.o_tov
    t["poss"] = 0.5 * (own + other)
    return t


def player_games(lg: str, y: int) -> pd.DataFrame:
    b = pd.read_parquet(ESPN / f"{lg}_player_box_{y}.parquet")
    b = b[b.season_type.isin(SEASON_TYPE) & (b.minutes.fillna(0) > 0)].copy()
    b = b.rename(columns=ESPN_COUNT_MAP)
    keep = ["game_id", "season", "season_type", "game_date", "athlete_id", "athlete_display_name",
            "athlete_position_abbreviation", "team_id", "team_abbreviation", "starter", "minutes",
            *COUNT_COLS]
    b = b[keep]
    # Drop exhibition teams (All-Star etc.): franchises play >= 10 regular-season games.
    n = b[b.season_type == 2].groupby("team_id").game_id.nunique()
    b = b[b.team_id.isin(n[n >= 10].index)]
    # Cup finals (NBA Cup / WNBA Commissioner's Cup) are kept as competitive games but
    # flagged: they do not count in the standings, so team-record analyses can drop them.
    b["game_context"] = b.season_type.map(SEASON_TYPE)
    ctx = game_context(lg, y)
    b.loc[b.game_id.isin(ctx.index), "game_context"] = b.game_id.map(ctx)
    tg = team_games(b, lg, y)
    team_min = b.groupby(["game_id", "team_id"]).minutes.sum().rename("team_min")
    b = b.join(team_min, on=["game_id", "team_id"]).merge(tg, on=["game_id", "team_id"], how="left")
    share = b.minutes / (b.team_min / 5.0)
    b["poss_on"] = b.poss * share
    b["orb_chances_on"] = (b.t_orb + b.o_drb) * share
    b["drb_chances_on"] = (b.t_drb + b.o_orb) * share
    b["tm_fgm_on"] = (b.t_fgm * share - b.fgm).clip(lower=0)
    b["opp_fga_on"] = b.o_fga * share
    b["league"] = lg.upper()
    return b


def game_context(lg: str, y: int) -> pd.Series:
    """game_id -> context from ESPN schedule notes: cup_group / cup_knockout / cup_final.
    Cup group and knockout games count in the standings; cup finals do not."""
    d = pd.read_parquet(ESPN / f"{lg}_schedule_{y}.parquet", columns=["game_id", "notes_headline", "type_abbreviation"])
    h = d.notes_headline.fillna("")
    cup = h.str.contains("Cup|In-Season Tournament", case=False)
    ctx = np.select(
        [(cup & h.str.contains("Championship")) | (d.type_abbreviation == "CC"),
         cup & h.str.contains("Quarterfinal|Semifinal"),
         cup],
        ["cup_final", "cup_knockout", "cup_group"], default="")
    return pd.Series(ctx, index=pd.to_numeric(d.game_id)).loc[lambda x: x != ""]


EXPO = ["poss_on", "orb_chances_on", "drb_chances_on", "tm_fgm_on", "opp_fga_on"]


def load_dob() -> pd.DataFrame:
    """Date of birth by (league, name key): ESPN rosters first, NBA player info as fallback."""
    rows = []
    for f in sorted(ESPN.glob("*_roster_*.parquet")):
        r = pd.read_parquet(f, columns=["athlete_id", "display_name", "date_of_birth"])
        r["league"] = f.name.split("_")[0].upper()
        rows.append(r)
    r = pd.concat(rows)
    r["dob"] = pd.to_datetime(r.date_of_birth, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    r["athlete_id"] = pd.to_numeric(r.athlete_id, errors="coerce")
    by_id = r.dropna(subset=["dob"]).drop_duplicates(["league", "athlete_id"])[["league", "athlete_id", "dob"]]
    cpi = pd.read_csv(next((RAW / "uploads").glob("*common_player_info.csv")),
                      usecols=["display_first_last", "birthdate", "from_year", "to_year"])
    cpi["key"] = cpi.display_first_last.map(norm_name)
    cpi["dob"] = pd.to_datetime(cpi.birthdate, errors="coerce")
    cpi = cpi.dropna(subset=["dob"])
    cpi = cpi[~cpi.key.duplicated(keep=False)]  # ambiguous names dropped, not guessed
    return by_id, cpi[["key", "dob", "from_year", "to_year"]]


def fill_integer_ages(ps: pd.DataFrame) -> None:
    """Fill remaining ages from sources that give whole-year age (Basketball-Reference based):
      WNBA <= 2019: FiveThirtyEight WNBA-stats 'Age' (as of Jul 1, same reference date);
      WNBA 2020-23: last FiveThirtyEight age + years since (players who debuted by 2019);
      NBA 2010-25: 'Age' in the user-supplied stats/salary file (as of Feb 1).
    Whole-year age is stored as age + 0.5 (expected value within the year)."""
    ps["age_source"] = np.where(ps.age.notna(), ps.dob_source, None)
    f = RAW / "external" / "538_wnba_player_stats.csv"
    if f.exists():
        w = pd.read_csv(f, usecols=["player_ID", "Player", "year_ID", "Age"])
        w["key"] = w.Player.map(norm_name)
        amb = w.groupby("key").player_ID.nunique()
        w = w[w.key.isin(amb[amb == 1].index)].drop_duplicates(["key", "year_ID"])
        by_season = w.set_index(["key", "year_ID"]).Age
        last = w.sort_values("year_ID").groupby("key").agg(y=("year_ID", "last"), a=("Age", "last"))
        m = ps.age.isna() & (ps.league == "WNBA")
        direct = pd.Series([by_season.get((k, s)) for k, s in zip(ps.key[m], ps.season[m])], index=ps.index[m], dtype=float)
        carry = pd.Series([last.a.get(k) + (s - last.y.get(k)) if (k in last.index and s > 2019 and s - last.y.get(k) <= 6) else np.nan
                           for k, s in zip(ps.key[m], ps.season[m])], index=ps.index[m], dtype=float)
        val = direct.fillna(carry)
        ps.loc[val.dropna().index, "age"] = val.dropna() + 0.5
        ps.loc[direct.dropna().index, "age_source"] = "538_bbref_age"
        ps.loc[carry.dropna().index.difference(direct.dropna().index), "age_source"] = "538_bbref_age_carried"
    sal = next((RAW / "uploads").glob("*NBA_Player_Stats_and_Salaries*.csv"), None)
    if sal is not None:
        s = pd.read_csv(sal, usecols=["Player", "Year", "Age"])
        s["key"] = s.Player.map(norm_name)
        s = s.drop_duplicates(["key", "Year"]).set_index(["key", "Year"]).Age
        m = ps.age.isna() & (ps.league == "NBA")
        val = pd.Series([s.get((k, y)) for k, y in zip(ps.key[m], ps.season[m])], index=ps.index[m], dtype=float).dropna()
        ps.loc[val.index, "age"] = val + 0.5
        ps.loc[val.index, "age_source"] = "user_nba_file_age"
    ps.loc[(ps.age < 18) | (ps.age > 45), ["age", "age_source"]] = [np.nan, "rejected_implausible"]


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    pg = pd.concat([player_games(lg, y) for lg in SEASONS for y in SEASONS[lg]], ignore_index=True)
    pg["season_type"] = pg.season_type.map(SEASON_TYPE)
    pg.to_parquet(PROC / "player_game.parquet", index=False)

    keys = ["league", "season", "season_type", "athlete_id"]
    agg = {c: "sum" for c in ["minutes", *COUNT_COLS, *EXPO]}
    agg.update(games=("game_id", "nunique"))
    # player-team-season (needed for team minutes shares in Step 5)
    pts = (pg.groupby([*keys, "team_id", "team_abbreviation"])
             .agg(**{k: (k, v) if isinstance(v, str) else v for k, v in agg.items()}).reset_index())
    pts.to_parquet(PROC / "player_team_season.parquet", index=False)

    meta = (pg.sort_values("game_date").groupby(keys)
              .agg(player=("athlete_display_name", "last"),
                   position=("athlete_position_abbreviation", lambda s: s.mode().iat[0] if len(s.mode()) else None),
                   teams=("team_abbreviation", lambda s: "|".join(dict.fromkeys(s))),
                   starts=("starter", "sum"), first_date=("game_date", "min"), last_date=("game_date", "max")))
    ps = (pg.groupby(keys).agg(**{k: (k, v) if isinstance(v, str) else v for k, v in agg.items()})
            .join(meta).reset_index())
    ps["position_raw"] = ps.position
    ps["position"] = ps.position.replace({"PG": "G", "SG": "G", "SF": "F", "PF": "F", "GF": "F", "NA": None})
    # Missing position: borrow the player's most common position in other seasons.
    mode = ps.dropna(subset=["position"]).groupby(["league", "athlete_id"]).position.agg(lambda s: s.mode().iat[0])
    miss = ps.position.isna()
    ps.loc[miss, "position"] = [mode.get((lg, a)) for lg, a in zip(ps.league[miss], ps.athlete_id[miss])]
    ps["position_imputed"] = miss & ps.position.notna()
    ps["n_teams"] = ps.teams.str.count(r"\|") + 1
    ps["atypical_season"] = [(lg.lower(), s) in ATYPICAL for lg, s in zip(ps.league, ps.season)]

    # Age at a fixed in-season reference date.
    by_id, cpi = load_dob()
    ps = ps.merge(by_id, on=["league", "athlete_id"], how="left")
    ps["key"] = ps.player.map(norm_name)
    ps["dob_source"] = np.where(ps.dob.notna(), "espn_roster", None)
    fill = ps[ps.dob.isna() & (ps.league == "NBA")].merge(cpi, on="key", how="inner")
    # Name-only match is accepted only if that person was active around this season
    # (player-info file ends in 2023, so later seasons need to_year >= 2022).
    active = (fill.from_year <= fill.season) & (fill.to_year >= np.minimum(fill.season - 1, 2022))
    fill = fill[active].set_index(["league", "season", "season_type", "athlete_id"]).dob_y
    idx = ps.set_index(["league", "season", "season_type", "athlete_id"]).index
    hit = idx.isin(fill.index) & ps.dob.isna().to_numpy()
    ps.loc[hit, "dob"] = fill.reindex(idx[hit]).to_numpy()
    ps.loc[hit, "dob_source"] = "nba_player_info_name_match"
    ref = pd.to_datetime([AGE_REF[lg.lower()].format(y=s) for lg, s in zip(ps.league, ps.season)])
    ps["age"] = ((ref - ps.dob).dt.days / 365.25).round(2)
    ps.loc[(ps.age < 18) | (ps.age > 45), ["age", "dob", "dob_source"]] = [np.nan, pd.NaT, "rejected_implausible"]
    fill_integer_ages(ps)
    ps.to_parquet(PROC / "player_season.parquet", index=False)
    print(f"player_game={len(pg):,} player_team_season={len(pts):,} player_season={len(ps):,}", file=sys.stderr)


if __name__ == "__main__":
    main()
