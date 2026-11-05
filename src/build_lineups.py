"""Rebuild the five players on the floor for each team from ESPN play-by-play, and aggregate stints.

Sources: play-by-play substitution events ("X enters the game for Y": athlete_id_1 in, athlete_id_2 out)
and box-score starters (opening five of each game). Player -> team comes from that game's box score.

Lineup at the start of periods 2+ is not in the data, so it is inferred per team:
  players who appear in the period (any event, or subbed OUT) before they are subbed IN were on the floor
  at the start; if fewer than 5 are found, fill from the previous period's closing five, then by minutes
  played; if more than 5, keep the 5 who appear earliest.
A substitution that removes a player not on the floor still adds the entering player; any moment where a
team does not have exactly 5 players is marked invalid and those events are dropped from the stints.

Per event, team stats: points (score change), field-goal attempts, free-throw attempts, offensive
rebounds, turnovers. Possessions per side = FGA + 0.44*FTA - ORB + TOV (the same estimate as Step 1).
Output: data/processed/stints/{league}_{season}.parquet, one row per (game, stint, offense side):
  off_team, def_team, off_1..off_5, def_1..def_5, poss, pts, seconds_proxy (events), season_type, home flag.
Validation: results/lineups_validation.csv compares stint totals with box-score team totals.
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import PROC, RAW, SEASONS, SEASON_TYPE

ESPN = RAW / "espn"
OUT = PROC / "stints"


def classify(tt: pd.Series, shooting: pd.Series, athlete: pd.Series) -> pd.DataFrame:
    """Team rebounds (no athlete) are excluded, as in box-score ORB, so possessions match Step 1."""
    t = tt.fillna("").str.lower()
    ft = t.str.startswith("free throw")
    return pd.DataFrame({
        "fga": shooting.fillna(False).astype(bool) & ~ft,
        "fta": ft,
        "orb": t.str.contains("offensive rebound") & athlete.notna(),
        "tov": (t.str.contains("turnover") & (t != "no turnover")) | t.isin(["traveling", "kicked ball", "lane violation turnover",
                                                    "offensive goaltending", "double dribble", "palming"]),
    })


def game_stints(g: pd.DataFrame, box: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    home, away = int(g.home_team_id.iat[0]), int(g.away_team_id.iat[0])
    team_of = dict(zip(box.athlete_id, box.team_id))
    mins = dict(zip(box.athlete_id, box.minutes.fillna(0)))
    starters = {t: list(box[(box.team_id == t) & box.starter].athlete_id) for t in (home, away)}
    g = g.sort_values(["period_number", "game_play_number"]).reset_index(drop=True)
    cls = classify(g.type_text, g.shooting_play, g.athlete_id_1)
    is_sub = (g.type_text == "Substitution").to_numpy()
    a1, a2, a3 = (g[c].to_numpy() for c in ("athlete_id_1", "athlete_id_2", "athlete_id_3"))
    per = g.period_number.to_numpy()
    lineup = {}
    closing = {t: starters[t] for t in (home, away)}
    rows_home, rows_away, valid = [], [], []
    stats = {"periods_inferred_ok": 0, "periods_filled": 0}
    for p in np.unique(per):
        idx = np.where(per == p)[0]
        for t in (home, away):
            if p == 1 and len(starters[t]) == 5:
                lineup[t] = set(starters[t])
                continue
            seen, entered = [], set()
            for i in idx:
                if is_sub[i]:
                    pin, pout = a1[i], a2[i]
                    if team_of.get(pin) == t and pin not in seen:
                        entered.add(pin)
                    if team_of.get(pout) == t and pout not in entered and pout not in seen:
                        seen.append(pout)
                else:
                    for x in (a1[i], a2[i], a3[i]):
                        if x == x and team_of.get(x) == t and x not in entered and x not in seen:
                            seen.append(x)
            cand = seen[:5]
            if len(cand) < 5:
                stats["periods_filled"] += 1
                for x in list(closing[t]) + sorted([k for k, v in team_of.items() if v == t], key=lambda k: -mins.get(k, 0)):
                    if len(cand) == 5:
                        break
                    if x not in cand and x not in entered:
                        cand.append(x)
            else:
                stats["periods_inferred_ok"] += 1
            lineup[t] = set(cand)
        for i in idx:
            if is_sub[i]:
                pin, pout = a1[i], a2[i]
                t = team_of.get(pin, team_of.get(pout))
                if t in lineup:
                    lineup[t].discard(pout)
                    if pin == pin:
                        lineup[t].add(pin)
            rows_home.append(tuple(sorted(lineup[home])))
            rows_away.append(tuple(sorted(lineup[away])))
            valid.append(len(lineup[home]) == 5 and len(lineup[away]) == 5)
        closing = {t: list(lineup[t]) for t in (home, away)}
    g["home_five"], g["away_five"], g["valid"] = rows_home, rows_away, valid
    # Points: scoreboard changes, unless they do not add up to the final score (the score fields reset in some
    # games, e.g. WNBA 2012); then each scoring play's own value (complete in those games, but missing on some
    # plays in WNBA 2018-19, where the scoreboard is fine).
    tm = g.team_id.to_numpy()
    hs, as_ = g.home_score.ffill().fillna(0).to_numpy(), g.away_score.ffill().fillna(0).to_numpy()
    dh, da = np.diff(np.r_[0, hs]).clip(min=0), np.diff(np.r_[0, as_]).clip(min=0)
    if abs(dh.sum() - hs.max()) <= 1 and abs(da.sum() - as_.max()) <= 1:
        g["home_pts"], g["away_pts"] = dh, da
    else:
        sv = np.where(g.scoring_play.astype(str).to_numpy() == "True", pd.to_numeric(g.score_value, errors="coerce").fillna(0).to_numpy(), 0)
        g["home_pts"] = np.where(tm == home, sv, 0)
        g["away_pts"] = np.where(tm == away, sv, 0)
    for c in ("fga", "fta", "orb", "tov"):
        g[f"home_{c}"] = cls[c].to_numpy() & (tm == home)
        g[f"away_{c}"] = cls[c].to_numpy() & (tm == away)
    g["stint"] = (g.home_five != g.home_five.shift()) | (g.away_five != g.away_five.shift())
    g["stint"] = g.stint.cumsum()
    stats["events"] = len(g)
    stats["valid_share"] = float(g.valid.mean())
    v = g[g.valid]
    agg = v.groupby("stint").agg(home_five=("home_five", "first"), away_five=("away_five", "first"), n_events=("valid", "size"),
                                 **{f"{s}_{c}": (f"{s}_{c}", "sum") for s in ("home", "away") for c in ("pts", "fga", "fta", "orb", "tov")})
    out = []
    for side, opp, off_t, def_t, is_home in (("home", "away", home, away, 1), ("away", "home", away, home, 0)):
        d = pd.DataFrame({
            "off_team": off_t, "def_team": def_t, "home": is_home, "n_events": agg.n_events,
            "pts": agg[f"{side}_pts"],
            "poss": agg[f"{side}_fga"] + 0.44 * agg[f"{side}_fta"] - agg[f"{side}_orb"] + agg[f"{side}_tov"],
        })
        for k in range(5):
            d[f"off_{k + 1}"] = [x[k] for x in agg[f"{side}_five"]]
            d[f"def_{k + 1}"] = [x[k] for x in agg[f"{opp}_five"]]
        out.append(d)
    return pd.concat(out, ignore_index=True), stats


def season(args) -> dict:
    lg, y = args
    f = ESPN / f"{lg}_pbp_{y}.parquet"
    if not f.exists():
        return {}
    pbp = pd.read_parquet(f, columns=["game_id", "season_type", "period_number", "game_play_number", "type_text", "team_id",
                                      "athlete_id_1", "athlete_id_2", "athlete_id_3", "home_score", "away_score",
                                      "shooting_play", "home_team_id", "away_team_id", "scoring_play", "score_value"])
    pbp = pbp[pbp.season_type.isin([2, 3])]
    box = pd.read_parquet(ESPN / f"{lg}_player_box_{y}.parquet", columns=["game_id", "athlete_id", "team_id", "starter", "minutes"])
    box["starter"] = box.starter.astype(str) == "True"
    box["minutes"] = pd.to_numeric(box.minutes, errors="coerce")
    boxes = dict(tuple(box.groupby("game_id")))
    res, st = [], []
    for gid, g in pbp.groupby("game_id"):
        if gid not in boxes:
            continue
        try:
            s, info = game_stints(g, boxes[gid])
        except Exception as e:  # malformed game: skip, and count it
            st.append({"game_id": gid, "error": str(e)[:80]})
            continue
        s.insert(0, "game_id", gid)
        s.insert(1, "season_type", SEASON_TYPE.get(int(g.season_type.iat[0]), "other"))
        res.append(s)
        st.append({"game_id": gid, **info})
    S = pd.concat(res, ignore_index=True)
    S.insert(0, "season", y)
    S.insert(0, "league", lg.upper())
    OUT.mkdir(parents=True, exist_ok=True)
    S.to_parquet(OUT / f"{lg.upper()}_{y}.parquet", index=False)
    st = pd.DataFrame(st)
    # validation vs box-score team totals (regular + playoffs)
    tb = box.groupby(["game_id", "team_id"]).size()  # presence only
    tot = S.groupby("game_id").agg(pts=("pts", "sum"), poss=("poss", "sum"))
    pg = pd.read_parquet(PROC / "player_game.parquet", columns=["league", "season", "game_id", "team_id", "t_pts", "poss"])
    pg = pg[(pg.league == lg.upper()) & (pg.season == y)].drop_duplicates(["game_id", "team_id"])
    ref = pg.groupby("game_id").agg(ref_pts=("t_pts", "sum"), ref_poss=("poss", "sum"))
    j = tot.join(ref, how="inner")
    return {"league": lg.upper(), "season": y, "games": len(st), "errors": int(st.get("error", pd.Series(dtype=str)).notna().sum()),
            "valid_event_share": float(st.valid_share.mean()), "period_start_inferred_share":
                float(st.periods_inferred_ok.sum() / (st.periods_inferred_ok.sum() + st.periods_filled.sum())),
            "pts_captured": float(j.pts.sum() / j.ref_pts.sum()), "poss_captured": float(j.poss.sum() / j.ref_poss.sum()),  # ref: one row per team, each = game possessions
            "stint_rows": len(S)}


def main(only=None) -> None:
    jobs = [(lg, y) for lg, ys in SEASONS.items() for y in ys]
    jobs = [j for j in jobs if only is None or j in only]
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = [r for r in ex.map(season, [(lg.lower(), y) for lg, y in jobs]) if r]
    v = pd.DataFrame(rows)
    path = PROC.parent.parent / "results" / "lineups_validation.csv"
    if only is not None and path.exists():
        old = pd.read_csv(path)
        v = pd.concat([old[~old.set_index(["league", "season"]).index.isin(v.set_index(["league", "season"]).index)], v])
    v.sort_values(["league", "season"]).to_csv(path, index=False)
    print(v.round(3).to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
