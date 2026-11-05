"""2026-27 NBA title odds from current rosters (ESPN roster file for season 2027, downloaded Nov 2026).

Roster-only model: no team results from 2026-27 exist yet, so team strength is built from players:
  o_prev, d_prev  sum over the roster of 5 x projected minutes share x player's 2025-26 offense / defense impact
                  (prior-informed lineup RAPM). Projected minutes = the player's 2025-26 regular-season minutes;
                  players with no 2025-26 NBA minutes (rookies, returners, imports) get 500 minutes at impact 0.
  age_w           projected-minutes-weighted roster age
Model: conditional logit (one champion per season among ALL teams), trained on NBA 2004-05 to 2025-26 with the same
  features built the same way (season s roster with s-1 impact and s-1 minutes as the projection). Its forward-chained
  record is reported next to the odds.
Roles: each player's 2025-26 archetype / sub-type. Players on a new team are flagged; from the trade analysis
  (Step 3), players who change teams change archetype about 29% of the time vs 17% for players who stay, so their
  listed role is less certain (e.g. Dončić kept his role after his trade, Bridges did not).
Outputs: results/step5_contenders_2027.csv, step5_contenders_cv.csv, step5_contenders_2027_players.csv
"""
import sys

import numpy as np
import pandas as pd

from common import PROC, RAW, RESULTS
from step5_models import clogit_fit, clogit_ll

FEATS = ["o_prev_z", "d_prev_z", "age_w_z"]
NEW_MIN = 500.0


def roster_features(roster: pd.DataFrame, season: int) -> pd.DataFrame:
    """roster: league, team_id, athlete_id, age. Uses season-1 impact and season-1 minutes as projection."""
    P = pd.read_parquet(PROC / "player_impact.parquet")
    P = P[(P.league == "NBA") & (P.season == season - 1)][["athlete_id", "o_impact", "d_impact", "impact"]]
    ps = pd.read_parquet(PROC / "player_season.parquet")
    ps = ps[(ps.league == "NBA") & (ps.season == season - 1) & (ps.season_type == "regular")][["athlete_id", "minutes"]]
    d = roster.merge(P, on="athlete_id", how="left").merge(ps, on="athlete_id", how="left")
    new = d.minutes.isna()
    d["proj_min"] = d.minutes.fillna(NEW_MIN)
    d[["o_impact", "d_impact", "impact"]] = d[["o_impact", "d_impact", "impact"]].fillna(0.0)
    d["new_to_nba_data"] = new

    def team(g):
        s = 5 * g.proj_min / g.proj_min.sum()
        return pd.Series({"o_prev": (s * g.o_impact).sum(), "d_prev": (s * g.d_impact).sum(),
                          "age_w": np.average(g.age.fillna(g.age.median()), weights=g.proj_min)})
    t = d.groupby("team_id").apply(team).reset_index()
    for c in ["o_prev", "d_prev", "age_w"]:
        t[c + "_z"] = (t[c] - t[c].mean()) / t[c].std(ddof=0)
    return t, d


def historical() -> pd.DataFrame:
    pt = pd.read_parquet(PROC / "player_team_season.parquet")
    pt = pt[(pt.league == "NBA") & (pt.season_type == "regular")]
    # one team per player-season: where he played the most minutes (end-of-season roster proxy)
    pt = pt.sort_values("minutes").drop_duplicates(["season", "athlete_id"], keep="last")
    ages = pd.read_parquet(PROC / "player_season_salary.parquet", columns=["league", "season", "athlete_id", "age"])
    ages = ages[ages.league == "NBA"]
    ts = pd.read_parquet(PROC / "team_seasons.parquet")
    ts = ts[ts.league == "NBA"][["season", "team_id", "team", "champion"]]
    rows = []
    for s in sorted(pt.season.unique())[1:]:
        r = pt[pt.season == s][["team_id", "athlete_id"]].merge(ages[ages.season == s][["athlete_id", "age"]], how="left")
        t, _ = roster_features(r, s)
        t["season"] = s
        rows.append(t)
    H = pd.concat(rows).merge(ts, on=["season", "team_id"])
    H["league"] = "NBA"
    H["champion"] = H.champion.astype(bool)
    return H


def main() -> None:
    H = historical()
    seasons = sorted(H.season.unique())
    cv = []
    for s in seasons[8:]:
        tr, te = H[H.season < s], H[H.season == s]
        if te.champion.sum() != 1:
            continue
        b = clogit_fit(tr, FEATS, lam=2.0)
        ll, pc = clogit_ll(te, FEATS, b)
        u = te[FEATS].to_numpy() @ b
        rank = int((u >= u[te.champion.to_numpy()][0]).sum())
        cv.append({"season": s, "champion": te[te.champion].team.iat[0], "prob_on_champion": pc, "champion_rank": rank})
    CV = pd.DataFrame(cv)
    CV.to_csv(RESULTS / "step5_contenders_cv.csv", index=False)
    b = clogit_fit(H, FEATS, lam=2.0)

    R = pd.read_parquet(RAW / "espn" / "nba_roster_2027.parquet")
    R = R.rename(columns={"display_name": "player"})[["team_id", "team_abbreviation", "team_display_name", "athlete_id", "player", "age"]]
    R["athlete_id"] = pd.to_numeric(R.athlete_id, errors="coerce")
    R["age"] = pd.to_numeric(R.age, errors="coerce")
    t, d = roster_features(R, 2027)
    u = t[FEATS].to_numpy() @ b
    p = np.exp(u - u.max())
    t["title_prob"] = p / p.sum()
    t = t.merge(R[["team_id", "team_abbreviation", "team_display_name"]].drop_duplicates(), on="team_id")

    # roles and team changes
    A = pd.read_parquet(PROC / "archetypes_detailed.parquet", columns=["league", "season", "athlete_id", "archetype", "subtype"])
    A = A[(A.league == "NBA") & (A.season == 2026)].drop(columns=["league", "season"])
    pt = pd.read_parquet(PROC / "player_team_season.parquet")
    last = pt[(pt.league == "NBA") & (pt.season == 2026) & (pt.season_type == "regular")].sort_values("first_date" if "first_date" in pt else "minutes")
    last = last.drop_duplicates("athlete_id", keep="last")[["athlete_id", "team_id"]].rename(columns={"team_id": "team_2026"})
    d = d.merge(A, on="athlete_id", how="left").merge(last, on="athlete_id", how="left")
    d["changed_team"] = d.team_2026.notna() & (d.team_2026 != d.team_id)
    d = d.merge(R[["athlete_id", "player", "team_abbreviation"]].drop_duplicates("athlete_id"), on="athlete_id", how="left",
                suffixes=("", "_r"))
    d["player"] = d.player.fillna(d.get("player_r"))

    def summary(g):
        g = g.sort_values("proj_min", ascending=False)
        top = g.head(3)
        arr = g[g.changed_team & (g.proj_min >= 1000)]
        return pd.Series({"top_players": ", ".join(f"{p} ({i:+.1f})" for p, i in zip(top.player, top.impact)),
                          "key_arrivals": ", ".join(arr.player.head(4)),
                          "main_roles": ", ".join(g.head(8).subtype.dropna().value_counts().head(3).index)})
    sm = d.groupby("team_id").apply(summary).reset_index()
    t = t.merge(sm, on="team_id").sort_values("title_prob", ascending=False)
    out = t[["team_display_name", "title_prob", "o_prev", "d_prev", "age_w", "top_players", "key_arrivals", "main_roles"]]
    out.round(4).to_csv(RESULTS / "step5_contenders_2027.csv", index=False)
    d[["team_abbreviation", "player", "proj_min", "o_impact", "d_impact", "impact", "archetype", "subtype", "changed_team",
       "new_to_nba_data"]].round(2).to_csv(RESULTS / "step5_contenders_2027_players.csv", index=False)
    print("forward-chained record:", CV.prob_on_champion.mean().round(3), "avg prob on champion;",
          (CV.champion_rank <= 3).mean().round(2), "top-3 rate over", len(CV), "seasons", file=sys.stderr)
    print(out.head(12).round(3).to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
