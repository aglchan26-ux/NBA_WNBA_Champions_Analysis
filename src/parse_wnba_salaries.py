"""Parse the pasted Her Hoop Stats 2026 WNBA salary sheet into a tidy CSV.

Input format: a player-name line followed by a tab-separated line that starts
with an optional salary ("$300,000 ") and then a signing/contract code.
Only salary and contract code are kept; stats come from ESPN box scores.
"""
import re
import sys
from pathlib import Path

import pandas as pd

import unicodedata


def norm(x: str) -> str:
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", " ", x))

# Signing codes seen on the sheet -> analysis cohort used in Step 6.
COHORT = {
    "Rookie": "rookie_scale",
    "Hardship": "short_term",
    "7D": "short_term",
    "ROS": "short_term",   # rest-of-season contract
    "UDFA": "standard",    # undrafted free agent (sheet code)
    "spotrac_fill": "unknown",  # amount from Spotrac; no signing code
    "Dev": "development",
    "Core": "core_max_tier",
    "UFA": "standard",
    "RFA": "standard",
    "Reserved": "standard",
    "UP": "standard",      # unprotected; treated as standard
    "SuspCE": "standard",  # suspended contract (sheet code); treated as standard
    "--": "unknown",
}


def parse(path: Path) -> pd.DataFrame:
    rows, name = [], None
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    if lines and lines[0].count("\t") == 2:  # compact format: player, salary, code
        for ln in lines:
            player, sal, code = (x.strip() for x in ln.split("\t"))
            rows.append({"player": player, "salary": float(sal) if sal else float("nan"), "signing_code": code})
        df = pd.DataFrame(rows)
        df["cohort"] = df.signing_code.map(COHORT).fillna("unknown")
        return df
    for line in lines:
        if "\t" not in line:
            name = line.strip()
            continue
        fields = [f.strip() for f in line.split("\t")]
        if fields[0].startswith("$"):
            salary = float(re.sub(r"[$,]", "", fields[0]))
            code = fields[1]
        else:
            salary, code = float("nan"), fields[0]
        rows.append({"player": name, "salary": salary, "signing_code": code})
    df = pd.DataFrame(rows)
    df["cohort"] = df.signing_code.map(COHORT).fillna("unknown")
    return df


def collapse(df: pd.DataFrame) -> pd.DataFrame:
    """One row per player: sum split contracts; cohort from the largest piece."""
    df = df.copy()
    df["_s"] = df.salary.fillna(-1)
    top = df.sort_values("_s", ascending=False).drop_duplicates("player")
    agg = df.groupby("player").agg(
        salary_total=("salary", lambda s: s.sum(min_count=1)),
        n_contract_rows=("salary", "size"),
        codes=("signing_code", lambda s: "|".join(s)),
    )
    out = agg.join(top.set_index("player")[["signing_code", "cohort"]])
    out["salary_missing_any"] = df.groupby("player").salary.apply(lambda s: s.isna().any())
    return out.reset_index()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = root / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    raws, cols = [], []
    for f in sorted((root / "data/raw/manual").glob("wnba_salaries_*_herhoopstats.tsv")):
        season = int(re.search(r"(\d{4})", f.name).group(1))
        raw = parse(f)
        raw.insert(0, "season", season)
        raw.insert(0, "league", "WNBA")
        col = collapse(raw)
        col.insert(0, "season", season)
        col.insert(0, "league", "WNBA")
        raws.append(raw)
        cols.append(col)
        print(f"{season}: rows={len(raw)} players={len(col)} salary_sum={raw.salary.sum():,.0f}", file=sys.stderr)
    # Supplements (e.g. Spotrac 2021): add ONLY players absent from that season's Her Hoop Stats sheet.
    for f in sorted((root / "data/raw/manual").glob("wnba_salaries_*_spotrac.tsv")):
        season = int(re.search(r"(\d{4})", f.name).group(1))
        sp = pd.read_csv(f, sep="\t", comment="#", names=["player", "team", "salary"])
        have = {norm(p) for r in raws if (r.season == season).any() for p in r[r.season == season].player}
        add = sp[~sp.player.map(norm).isin(have)].copy()
        add = add.assign(league="WNBA", season=season, signing_code="spotrac_fill", cohort="unknown")[
            ["league", "season", "player", "salary", "signing_code", "cohort"]]
        col = collapse(add.drop(columns=["league", "season"]))
        col.insert(0, "season", season)
        col.insert(0, "league", "WNBA")
        raws.append(add)
        cols.append(col)
        print(f"{season}: +{len(add)} players from {f.name} (absent from Her Hoop Stats)", file=sys.stderr)
    pd.concat(raws).to_csv(out / "wnba_salary_rows.csv", index=False)
    pd.concat(cols).to_csv(out / "wnba_salary_players.csv", index=False)
