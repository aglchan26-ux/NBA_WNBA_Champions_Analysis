"""Step 5a: team-season roster profiles, playoff results, and the 'championship checklist'.

All roster features are regular-season, minutes-weighted, built from player layers (Step 3).
Team minutes share of a player = player minutes / (team games x game length x 5).

Playoff results: series = all playoff games between the same two teams in a season; the team
with more wins takes the series; rounds_won counts series won; the champion wins the series
that starts last. (WNBA single-game rounds in some years count as series of length 1.)

Checklist (owner's championship formula, tested descriptively - association only):
  C1 primary scorer      a player with >= 9% of team minutes who is an Alpha scorer (owner's definition:
                         top usage + scoring, any style) or a Shot-making scorer with usage >= 27%
  C2 secondary scorer    a second player with >= 9% of team minutes and usage >= 23%
  C3 floor-spacing big   >= 9% of team minutes from bigs (C, or F in big/forward archetypes, or
                         'Stretch big (spot-up)') whose three-point share is >= 0.5 SD above other
                         players at their position that league-season
  C4 perimeter defender  >= 9% of team minutes from guards/wings (G, or F outside big archetypes) with
                         perimeter_d_index >= 0.5 (steals, jump-shot blocks, deflections, contested
                         3s where recorded). Seasons without event detail fall back to the defensive
                         sub-types (Perimeter stopper forward, Defensive/connector guard, glue defender).
  Clash                  number of players with >= 9% of team minutes and usage >= 27%
(9% of team minutes ~ 22 minutes a game in an 82-game NBA season.)
"""
import sys

import numpy as np
import pandas as pd

from common import PROC, RESULTS

ROLE_MIN = 0.09
PRIMARY = {"Alpha scorer", "Shot-making scorer", "Point forward / scoring hub"}
PERIMETER_D = {"Perimeter stopper forward", "Defensive / connector guard", "Low-usage glue defender", "Floor general"}
BIG_ARCH = {"Hub big / point forward", "Rim-running big", "Connector forward"}
RIM_D = {"Defensive anchor big", "Low-usage rim protector", "Mobile defensive big", "Rim-defending big forward"}


def series_results() -> pd.DataFrame:
    pg = pd.read_parquet(PROC / "player_game.parquet", columns=["league", "season", "season_type", "game_id", "game_date",
                                                                "team_id", "team_abbreviation", "t_pts", "o_pts"])
    g = pg[pg.season_type == "playoffs"].drop_duplicates(["game_id", "team_id"]).copy()
    g["win"] = (g.t_pts > g.o_pts).astype(int)
    opp = g[["game_id", "team_id"]].rename(columns={"team_id": "opp_id"})
    g = g.merge(opp, on="game_id")
    g = g[g.team_id != g.opp_id]
    g["pair"] = [tuple(sorted(p)) for p in zip(g.team_id, g.opp_id)]
    s = g.groupby(["league", "season", "pair", "team_id"]).agg(wins=("win", "sum"), start=("game_date", "min"),
                                                               team=("team_abbreviation", "first")).reset_index()
    s["won_series"] = s.groupby(["league", "season", "pair"]).wins.transform(lambda w: (w == w.max()) & (w.max() > w.min()))
    last = s.groupby(["league", "season"]).start.transform("max")
    s["final"] = s.start == last
    out = s.groupby(["league", "season", "team_id"]).agg(team=("team", "first"), series_played=("pair", "nunique"),
                                                         rounds_won=("won_series", "sum"),
                                                         champion=("won_series", lambda x: False)).reset_index()
    champ = s[s.final & s.won_series][["league", "season", "team_id"]]
    out = out.merge(champ.assign(champion=True), on=["league", "season", "team_id"], how="left", suffixes=("_x", ""))
    out["champion"] = out.champion.fillna(False).astype(bool)
    out["finalist"] = out.merge(s[s.final][["league", "season", "team_id"]].drop_duplicates().assign(f=True),
                                on=["league", "season", "team_id"], how="left").f.fillna(False).astype(bool).to_numpy()
    return out.drop(columns=[c for c in out.columns if c.endswith("_x")])


def team_ratings() -> pd.DataFrame:
    pg = pd.read_parquet(PROC / "player_game.parquet", columns=["league", "season", "season_type", "game_context", "game_id",
                                                                "team_id", "team_abbreviation", "poss", "t_pts", "o_pts"])
    g = pg[(pg.season_type == "regular") & (pg.game_context != "cup_final")].drop_duplicates(["game_id", "team_id"])
    r = g.groupby(["league", "season", "team_id"]).agg(team=("team_abbreviation", "first"), games=("game_id", "nunique"),
                                                       poss=("poss", "sum"), pts=("t_pts", "sum"),
                                                       opp_pts=("o_pts", "sum")).reset_index()
    w = g.assign(w=(g.t_pts > g.o_pts).astype(int)).groupby(["league", "season", "team_id"]).w.sum()
    r["wins"] = w.to_numpy()
    r["ortg"], r["drtg"] = 100 * r.pts / r.poss, 100 * r.opp_pts / r.poss
    r["net_rating"] = r.ortg - r.drtg
    r["win_pct"] = r.wins / r.games
    return r


def roster_features() -> pd.DataFrame:
    L = pd.read_parquet(PROC / "player_layers.parquet")
    pts = pd.read_parquet(PROC / "player_team_season.parquet")
    pts = pts[pts.season_type == "regular"][["league", "season", "athlete_id", "team_id", "minutes"]].rename(
        columns={"minutes": "team_minutes_played"})
    tr = team_ratings()[["league", "season", "team_id", "games"]]
    d = pts.merge(L.drop(columns=["minutes"]), on=["league", "season", "athlete_id"], how="left").merge(
        tr, on=["league", "season", "team_id"])
    glen = np.where(d.league == "NBA", 48.0, 40.0)
    d["msh"] = d.team_minutes_played / (d.games * glen * 5)
    sh = pd.read_parquet(PROC / "player_season_shrunk.parquet", columns=["league", "season", "athlete_id", "usage_eb", "fg3_pct_eb"])
    d = d.merge(sh, on=["league", "season", "athlete_id"], how="left")
    reg = d.msh >= ROLE_MIN
    d["is_primary"] = reg & (d.alpha_scorer.fillna(False) | ((d.subtype == "Shot-making scorer") & (d.usage_eb >= 0.27)))
    d["is_high_usage"] = reg & (d.usage_eb >= 0.27)
    d["is_secondary"] = reg & (d.usage_eb >= 0.23)
    big = (d.position == "C") | ((d.position == "F") & d.archetype.isin(BIG_ARCH)) | (d.subtype == "Stretch big (spot-up)")
    d["is_spacing_big"] = big & (d.three_diet_pos_z >= 0.5)
    perim = (d.position == "G") | ((d.position == "F") & ~d.archetype.isin(BIG_ARCH))
    has_events = d.perimeter_d_index.notna()
    d["is_perim_d"] = perim & np.where(has_events, d.perimeter_d_index >= 0.5, d.style_subtype.isin(PERIMETER_D))
    d["is_rim_d"] = d.subtype.isin(RIM_D)
    rows = []
    for (lg, s, t), g in d.groupby(["league", "season", "team_id"]):
        m = g.msh
        tot = m.sum()
        u = g.usage_eb.fillna(g.usage_eb.mean())
        r = {"league": lg, "season": s, "team_id": t,
             "C1_primary_scorer": bool(g.is_primary.any()),
             "C2_secondary_scorer": int(g.is_secondary.sum()) >= 2,
             "C3_spacing_big_share": float(m[g.is_spacing_big].sum() / tot),
             "C4_perimeter_d_share": float(m[g.is_perim_d].sum() / tot),
             "rim_protector_share": float(m[g.is_rim_d].sum() / tot),
             "n_high_usage": int(g.is_high_usage.sum()),
             "top_usage": float(u[reg].max()) if reg.any() else np.nan,
             "usage_concentration": float(((m * u) / (m * u).sum()).pow(2).sum()),
             "team_three_diet": float(np.average(g.three_diet.fillna(g.three_diet.mean()), weights=m)),
             "team_rim_diet": float(np.average(g.rim_diet.fillna(g.rim_diet.mean()), weights=m)),
             "team_mid_diet": float(np.average(g.mid_diet.fillna(g.mid_diet.mean()), weights=m)),
             "eff_vs_role_weighted": float(np.average(g.eff_vs_role.fillna(0), weights=m)),
             "shooter_share": float(m[(g.three_diet_z >= 0) & (g.fg3_pct_eb >= g.fg3_pct_eb.median())].sum() / tot)}
        for a, h in g.groupby("archetype"):
            r[f"arch_share::{a}"] = float(h.msh.sum() / tot)
        for a, h in g.groupby("subtype"):
            r[f"sub_share::{a}"] = float(h.msh.sum() / tot)
        if g.defense_tags.ne("n/a (season not covered)").any():
            r["def_rim_block_z"] = float(np.average(g.blk_rim_z.fillna(0), weights=m))
            r["def_steal_z"] = float(np.average((g.stl_pass_z.fillna(0) + g.stl_onball_z.fillna(0)) / 2, weights=m))
        rows.append(r)
    f = pd.DataFrame(rows)
    share_cols = [c for c in f.columns if c.startswith(("arch_share::", "sub_share::"))]
    f[share_cols] = f[share_cols].fillna(0.0)
    f["C3_spacing_big"] = f.C3_spacing_big_share >= ROLE_MIN
    f["C4_perimeter_d"] = f.C4_perimeter_d_share >= ROLE_MIN
    f["checklist_score"] = f[["C1_primary_scorer", "C2_secondary_scorer", "C3_spacing_big", "C4_perimeter_d"]].sum(axis=1)
    return f


def main() -> None:
    T = team_ratings().merge(roster_features(), on=["league", "season", "team_id"]).merge(
        series_results().drop(columns="team"), on=["league", "season", "team_id"], how="left")
    T["made_playoffs"] = T.series_played.notna()
    for c in ["series_played", "rounds_won"]:
        T[c] = T[c].fillna(0)
    T["champion"] = T.champion.fillna(False).astype(bool)
    T["finalist"] = T.finalist.fillna(False).astype(bool)
    # Results missing from the ESPN release (documented, sourced overrides; see results_overrides.csv)
    from common import RAW
    ov = RAW / "manual" / "results_overrides.csv"
    T["result_source"] = "espn_play_by_play"
    if ov.exists():
        for _, o in pd.read_csv(ov).iterrows():
            m = (T.league == o.league) & (T.season == o.season) & (T.team == o.team)
            if o.event == "champion" and m.any():
                T.loc[m, "champion"] = True
                T.loc[m, "rounds_won"] = T.loc[m, "rounds_won"] + 1
                T.loc[m, "result_source"] = "owner_override"
    T.to_parquet(PROC / "team_seasons.parquet", index=False)
    chk = T.groupby(["league", "season"]).champion.sum()
    print("seasons with exactly one champion:", int((chk == 1).sum()), "of", len(chk), file=sys.stderr)
    print(T[T.champion][["league", "season", "team", "net_rating", "checklist_score"]].tail(12).to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
