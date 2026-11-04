"""Parse the pasted Basketball-Reference 2026-27 NBA contracts table (see the .SOURCE.txt next to it).

One row per player for season 2027 (2026-27). The salary columns repeat the full amount when two teams
owe the player money, so the amount is taken once; all teams and per-team guaranteed money are kept.
Exact duplicate rows (same player, team and numbers) are dropped. No amount is imputed.
"""
import sys

import pandas as pd

from common import PROC, RAW, norm_name


def main() -> None:
    f = RAW / "manual" / "nba_contracts_2026_27_bbref.csv"
    d = pd.read_csv(f, header=1).rename(columns={"-9999": "bbref_id", "Tm": "team", "Player": "player"})
    money = [c for c in d.columns if c[:2] == "20" or c == "Guaranteed"]
    for c in money:
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    n0 = len(d)
    d = d.drop_duplicates(subset=[c for c in d.columns if c != "Rk"])
    g = d.groupby("bbref_id")
    out = g.agg(player=("player", "first"), teams=("team", "|".join), n_team_rows=("team", "size"),
                salary_2027=("2026-27", "first"), salary_2028=("2027-28", "first"),
                guaranteed_total=("Guaranteed", lambda s: s.sum(min_count=1))).reset_index()
    out["salary_consistent"] = g["2026-27"].nunique().values == 1
    out["key"] = out.player.map(norm_name)
    out.insert(0, "season", 2027)
    out.insert(0, "league", "NBA")
    out.to_csv(PROC / "nba_contracts_2027.csv", index=False)
    print(f"rows={n0} after exact-dup drop={len(d)} players={len(out)} "
          f"multi-team={int((out.n_team_rows > 1).sum())} inconsistent={int((~out.salary_consistent).sum())} "
          f"sum_2027=${out.salary_2027.sum():,.0f}", file=sys.stderr)


if __name__ == "__main__":
    main()
