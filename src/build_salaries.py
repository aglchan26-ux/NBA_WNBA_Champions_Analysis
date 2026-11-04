"""Step 1c: attach salary and contract information to regular-season player-seasons.

NBA: two user-supplied files (2000-2025 and 2010-2025; identical where they overlap),
     season = end year, same convention as ESPN. Joined on normalized name + season. No contract type in the
     source; a rookie-scale flag is derived from draft_history.csv (first-round pick, within
     the first four seasons after the draft). Minimum / max cohorts are assigned in Step 6
     from cited CBA figures.
     2025-26 (season 2026): ESPN salary list ranks 1-475 pasted by the owner (full list).
WNBA: Her Hoop Stats salary sheets 2025 and 2026 (pasted). Signing code -> cohort mapping in
     parse_wnba_salaries.py.
Unmatched rows keep salary = NaN with salary_status explaining why; nothing is imputed.
"""
import sys

import numpy as np
import pandas as pd

from common import PROC, RAW, norm_name


def nba_salaries() -> pd.DataFrame:
    """Union of the two user-supplied NBA salary files (same values on all 7,296 overlapping
    player-seasons): NBA_Player_Salaries_2000-2025.csv (Season = season end year) and
    NBA_Player_Stats_and_Salaries_2010-2025.csv (Year = season end year)."""
    up = RAW / "uploads"
    a = pd.read_csv(next(up.glob("*NBA_Player_Salaries_2000-2025*.csv"))).rename(
        columns={"Player": "Player", "Salary": "salary", "Season": "season"})
    a["salary_source"] = "nba_salaries_2000_2025"
    a["salary_team"] = None
    b = pd.read_csv(next(up.glob("*NBA_Player_Stats_and_Salaries*.csv")), usecols=["Player", "Salary", "Year", "Team"]).rename(
        columns={"Salary": "salary", "Year": "season", "Team": "salary_team"})
    b["salary_source"] = "nba_stats_and_salaries_2010_2025"
    c = pd.read_csv(RAW / "manual" / "nba_salaries_2026_espn.tsv", sep="\t", comment="#",
                    names=["Player", "pos", "salary_team", "salary"]).drop(columns="pos")
    c["season"] = 2026  # ESPN 2025-26 list, ranks 1-475 (pasted by owner)
    c["salary_source"] = "espn_2025_26"
    s = pd.concat([b, a, c], ignore_index=True)
    s["key"] = s.Player.map(norm_name)
    return s.drop_duplicates(["key", "season"], keep="first")


def rookie_flags() -> pd.DataFrame:
    d = pd.read_csv(next((RAW / "uploads").glob("*draft_history.csv")),
                    usecols=["player_name", "season", "round_number", "overall_pick"])
    d["key"] = d.player_name.map(norm_name)
    d = d[~d.key.duplicated(keep=False)]  # ambiguous names are not guessed
    return d.rename(columns={"season": "draft_year"})[["key", "draft_year", "round_number", "overall_pick"]]


def main() -> None:
    ps = pd.read_parquet(PROC / "player_season.parquet")
    ps["key"] = ps.player.map(norm_name)
    reg = ps[ps.season_type == "regular"].copy()

    nba = reg[reg.league == "NBA"].merge(nba_salaries()[["key", "season", "salary", "salary_team", "salary_source"]],
                                         on=["key", "season"], how="left")
    nba = nba.merge(rookie_flags(), on="key", how="left")
    fb_path = RAW / "external" / "nba_salaries_master.csv"
    if fb_path.exists():  # flagged fallback only; never overwrites the user's files
        fb = pd.read_csv(fb_path)
        fb["key"] = fb.player.map(norm_name)
        fb = fb.groupby(["key", "year"]).salary.sum().rename("salary_fallback").reset_index().rename(columns={"year": "season"})
        nba = nba.merge(fb, on=["key", "season"], how="left")
        nba.loc[nba.salary.notna(), "salary_fallback"] = np.nan
    yrs = nba.season - nba.draft_year  # NBA season ending in draft_year+1 is year 1
    nba["rookie_scale"] = (nba.round_number == 1) & yrs.between(1, 4)
    nba["contract_code"] = np.where(nba.rookie_scale, "rookie_scale_derived", None)
    nba["salary_status"] = np.select(
        [(nba.season > 2026) | (nba.season < 2000), nba.salary.notna()],
        ["no_source_for_season", "matched"], default="missing_in_source")

    w = pd.read_csv(PROC / "wnba_salary_players.csv")
    w["key"] = w.player.map(norm_name)
    # Name variants between ESPN and the salary sheets (married / hyphenated surnames, e.g.
    # 'Skylar Diggins-Smith' vs 'Skylar Diggins'): map an ESPN key to a sheet key of the same season
    # when one starts with the other, the shorter has >= 10 letters and the match is unique.
    espn_keys = reg[reg.league == "WNBA"][["season", "key"]].drop_duplicates()
    alias = {}
    for s_, g in espn_keys.groupby("season"):
        sheet = set(w[w.season == s_].key)
        for k in set(g.key) - sheet:
            c = [x for x in sheet if (x.startswith(k) or k.startswith(x)) and min(len(x), len(k)) >= 10]
            if len(c) == 1:
                alias[(s_, k)] = c[0]
    reg["key"] = [alias.get((s_, k), k) if lg == "WNBA" else k for lg, s_, k in zip(reg.league, reg.season, reg.key)]
    pd.Series({f"{k[0]}:{k[1]}": v for k, v in alias.items()}, name="sheet_key").to_csv(
        PROC.parent.parent / "results" / "wnba_salary_name_aliases.csv")
    wn = reg[reg.league == "WNBA"].merge(
        w[["key", "season", "salary_total", "signing_code", "cohort", "salary_missing_any"]]
        .rename(columns={"salary_total": "salary", "signing_code": "contract_code"}),
        on=["key", "season"], how="left")
    wn["salary_status"] = np.select(
        [~wn.season.isin(w.season.unique()), wn.salary.notna(), wn.contract_code.notna()],
        ["no_source_for_season", "matched", "listed_without_salary"], default="not_on_sheet")

    out = pd.concat([nba, wn], ignore_index=True)
    out.to_parquet(PROC / "player_season_salary.parquet", index=False)
    rep = (out[out.salary_status != "no_source_for_season"]
           .groupby(["league", "season"])
           .apply(lambda d: pd.Series({
               "players": len(d), "salary_matched": d.salary.notna().mean(),
               "minutes_weighted_matched": (d.minutes * d.salary.notna()).sum() / d.minutes.sum()}))
           .round(3))
    rep.to_csv(PROC.parent.parent / "results" / "step1_salary_coverage.csv")
    print(rep.to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
