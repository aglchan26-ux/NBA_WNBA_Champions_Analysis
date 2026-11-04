"""Export blended / limited-sample player-seasons for review, plus a blind hand-labeling sheet.

Outputs (labeling/):
  archetype_review.xlsx   all blended and limited-sample player-seasons WITH model labels
  hand_label_blind.xlsx   stratified sample WITHOUT model labels (for the gold-standard check)
  results/hand_label_key.csv  sample_id -> player-season (kept out of the blind sheet)
"""
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from common import PROC, RESULTS, ROOT

OUT = ROOT / "labeling"
YELLOW = PatternFill("solid", fgColor="FFFF00")
HEAD = PatternFill("solid", fgColor="D9E1F2")
FONT = Font(name="Arial", size=10)
BOLD = Font(name="Arial", size=10, bold=True)

ARCH = [
    ("On-ball creator", "Alpha scorer", "Highest usage, scores and creates, rebounds/blocks above guard norms", "Edwards, Banchero, Durant; Clark"),
    ("On-ball creator", "Shot-making scorer", "High usage, unassisted shot-making, little defense/rebounding", "DeRozan, Booker, LaVine; Plum, Bueckers"),
    ("On-ball creator", "Floor general", "Creator who passes more than scores, steals", "Haliburton, Fox, VanVleet; C. Gray"),
    ("Secondary guard", "Microwave scoring guard", "Moderate-high usage scorer, low assists, 3s and long mid", "Maxey, Quickley; K. Mitchell, R. Howard"),
    ("Secondary guard", "Defensive / connector guard", "Lower usage, assists and steals, little scoring", "D. White, Smart, CP3; Cloud, Vandersloot"),
    ("Spot-up 3-and-D wing", "Movement shooter", "Low usage, high 3P volume and accuracy, few defensive events", "G. Allen, Strus, Beasley"),
    ("Spot-up 3-and-D wing", "Catch-and-shoot wing", "Catch-and-shoot 3s; hustle data shows little extra defensive activity", "Camara, K. Murray, D. Brooks; McBride"),
    ("Spot-up 3-and-D wing", "Low-usage glue defender", "Lowest usage, corner 3s, blocks/boards for a wing", "KCP, Dort, O'Neale"),
    ("Connector forward", "Scoring stretch forward", "Forward with above-average usage and 3s", "MPJ, H. Barnes; A. Gray"),
    ("Connector forward", "Perimeter stopper forward", "Forward with high steals, low usage", "Anunoby, McDaniels, Herb Jones; Carrington"),
    ("Connector forward", "Rim-defending big forward", "Low-usage forward with blocks and offensive boards", "B. Lopez, A. Gordon; Bonner"),
    ("Hub big / point forward", "Playmaking hub", "High-usage big who passes, steals", "Jokic, Sabonis; A. Thomas, Stewart"),
    ("Hub big / point forward", "Stretch big", "High-usage big who shoots 3s well", "Towns, Vucevic; Jonquel Jones"),
    ("Hub big / point forward", "Defensive anchor big", "High-usage big, blocks, no 3s", "A. Davis, Mobley, Adebayo; Boston"),
    ("Rim-running big", "Rim-running finisher", "Rim shots and boards, moderate usage", "Zubac, J. Allen, Claxton; Magbegor"),
    ("Rim-running big", "Low-usage rim protector", "Very low usage, blocks and boards", "Gobert, Capela, Adams, Looney"),
    ("Hub big / point forward", "Jumbo playmaker", "Listed guard who plays like a playmaking big (position-adjusted label)", "B. Simmons, J. Giddey, A. Thompson"),
    ("(any guard/wing archetype)", "Stretch big (spot-up)", "Listed center whose style matches a spot-up wing (position-adjusted label)", "B. Lopez 2018-19, Horford, M. Bonner"),
    ("(other)", "Blended / tweener", "Fits between archetypes - use the notes column to say which", ""),
    ("(other)", "Can't tell", "Not enough information", ""),
]


def player_info() -> pd.DataFrame:
    A = pd.read_parquet(PROC / "archetypes_detailed.parquet")
    ps = pd.read_parquet(PROC / "player_season.parquet")
    cnt = ["teams", "age", "pts", "orb", "drb", "ast", "stl", "blk", "tov", "fga", "fg3a", "fta", "fgm", "fg3m", "poss_on"]
    ps = ps[ps.season_type == "regular"][["league", "season", "athlete_id", *cnt]].rename(columns={c: f"c_{c}" for c in cnt})
    sh = pd.read_parquet(PROC / "player_season_shrunk.parquet")[["league", "season", "athlete_id", "rim_share_eb", "fg3_rate_eb"]]
    d = A.merge(ps, on=["league", "season", "athlete_id"], how="left").merge(sh, on=["league", "season", "athlete_id"], how="left")
    g = d.games.clip(lower=1)
    probs = [c for c in d.columns if c.startswith("prob_")]
    P = d[probs].to_numpy()
    order = np.argsort(-P, axis=1)
    names = np.array([c[5:] for c in probs])
    season_label = np.where(d.league == "NBA", (d.season - 1).astype(str) + "-" + (d.season % 100).astype(str).str.zfill(2),
                            d.season.astype(str))
    out = pd.DataFrame({
        "league": d.league, "season": season_label, "player": d.player, "teams": d.c_teams, "position": d.position,
        "age": d.c_age.round(1), "games": d.games, "starts": d.starts, "minutes": d.minutes.round(0),
        "mpg": (d.minutes / g).round(1), "role_tier": d.role_tier,
        "pts_pg": (d.c_pts / g).round(1), "reb_pg": ((d.c_orb + d.c_drb) / g).round(1), "ast_pg": (d.c_ast / g).round(1),
        "stl_pg": (d.c_stl / g).round(1), "blk_pg": (d.c_blk / g).round(1), "tov_pg": (d.c_tov / g).round(1),
        "fg_pct": (d.c_fgm / d.c_fga.replace(0, np.nan)).round(3), "fg3a_pg": (d.c_fg3a / g).round(1),
        "fg3_pct": (d.c_fg3m / d.c_fg3a.replace(0, np.nan)).round(3),
        "usage_pct": (100 * (d.c_fga + 0.44 * d.c_fta + d.c_tov) / d.c_poss_on).round(1),
        "rim_shot_share_shrunk": d.rim_share_eb.round(3), "three_rate_shrunk": d.fg3_rate_eb.round(3),
        "definition": d.definition, "model_archetype": d.archetype, "model_subtype": d.subtype,
        "position_adjusted": d.position_adjusted,
        "own_membership": d.own_membership.round(3),
        "top1_archetype": names[order[:, 0]], "top1_prob": P[np.arange(len(P)), order[:, 0]].round(3),
        "top2_archetype": names[order[:, 1]], "top2_prob": P[np.arange(len(P)), order[:, 1]].round(3),
        "_season_num": d.season, "_athlete_id": d.athlete_id,
    })
    return out


def style(ws, label_cols: list, widths: dict | None = None) -> None:
    for c in ws[1]:
        c.font, c.fill = BOLD, HEAD
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font = FONT
    hdr = [c.value for c in ws[1]]
    for name in label_cols:
        j = hdr.index(name) + 1
        for r in range(2, ws.max_row + 1):
            ws.cell(r, j).fill = YELLOW
    for j, h in enumerate(hdr, 1):
        ws.column_dimensions[get_column_letter(j)].width = (widths or {}).get(h, max(9, min(26, len(str(h)) + 2)))
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = ws.dimensions


def add_validation(ws, col_name: str, n_rows: int, list_ref: str) -> None:
    j = [c.value for c in ws[1]].index(col_name) + 1
    dv = DataValidation(type="list", formula1=list_ref, allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{get_column_letter(j)}2:{get_column_letter(j)}{n_rows + 1}")


def write_lists(wb) -> str:
    ws = wb.create_sheet("Archetype list")
    ws.append(["archetype", "subtype", "defining traits (vs league-season average)", "recent examples (NBA; WNBA)"])
    for r in ARCH:
        ws.append(list(r))
    style(ws, [], {"archetype": 24, "subtype": 28, "defining traits (vs league-season average)": 60,
                   "recent examples (NBA; WNBA)": 42})
    ws.freeze_panes = "A2"
    return f"'Archetype list'!$B$2:$B${len(ARCH) + 1}"


LABEL_COLS = ["hand_subtype", "hand_confidence (1 = most sure, 3 = least sure)", "hand_notes"]


def review_workbook(info: pd.DataFrame) -> None:
    path = OUT / "archetype_review.xlsx"
    cols = [c for c in info.columns if not c.startswith("_")]
    sheets = {"Blended": info[info.definition == "blended"], "Limited sample": info[info.definition == "limited_sample"]}
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame({"x": []}).to_excel(xw, sheet_name="README", index=False)
        for name, df in sheets.items():
            df = df.sort_values(["league", "_season_num", "minutes"], ascending=[True, False, False])
            df = df[cols].assign(**{c: "" for c in LABEL_COLS})
            df.to_excel(xw, sheet_name=name, index=False)
    wb = load_workbook(path)
    ref = write_lists(wb)
    for name, df in sheets.items():
        ws = wb[name]
        style(ws, LABEL_COLS, {"player": 22, "teams": 12, "model_archetype": 22, "model_subtype": 26,
                               "top1_archetype": 22, "top2_archetype": 22, "hand_subtype": 26, "hand_notes": 30})
        add_validation(ws, "hand_subtype", len(df), ref)
    rd = wb["README"]
    rd.delete_rows(1, rd.max_row)
    lines = [
        ("Archetype review: blended and limited-sample player-seasons", True),
        ("Model labels are shown here. If your labels will be used to CHECK the model, label the blind sheet "
         "(hand_label_blind.xlsx) instead, so seeing the model's answer cannot sway you.", False),
        ("", False),
        ("Fill in only the yellow columns: hand_subtype (dropdown from 'Archetype list'), hand_confidence (1 = most "
         "sure, 3 = least sure), hand_notes.", False),
        ("Definitions", True),
        ("blended: >= 500 minutes but < 50% probability of belonging to its own model archetype (sits between types).", False),
        ("limited_sample: < 250 season minutes; traits are pulled toward the position average by shrinkage.", False),
        ("own_membership / top1 / top2: Gaussian-mixture probabilities over the six archetypes.", False),
        ("usage_pct = 100 x (FGA + 0.44 FTA + TOV) / possessions on floor. *_shrunk = Step 2 empirical-Bayes estimates.", False),
        ("Source: ESPN box scores and play-by-play via sportsdataverse; ages from ESPN rosters, FiveThirtyEight "
         "WNBA-stats (Basketball-Reference) and the user's NBA stats file.", False),
        ("Counts", True),
    ]
    for text, bold in lines:
        rd.append([text])
        rd.cell(rd.max_row, 1).font = BOLD if bold else FONT
    for name in sheets:
        rd.append([f"{name}: rows", f"=COUNTA('{name}'!C:C)-1"])
        rd.append([f"{name}: rows hand-labeled", f"=COUNTA('{name}'!{get_column_letter(cols.index('top2_prob') + 2)}:"
                                                 f"{get_column_letter(cols.index('top2_prob') + 2)})-1"])
    for r in rd.iter_rows(min_row=rd.max_row - 3):
        for c in r:
            c.font = FONT
    rd.column_dimensions["A"].width = 110
    rd.column_dimensions["B"].width = 12
    wb.move_sheet("Archetype list", offset=-(len(wb.sheetnames) - 2))
    wb.save(path)


def blind_workbook(info: pd.DataFrame, n_per_cell: dict, name: str = "hand_label_blind.xlsx",
                   key_name: str = "hand_label_key.csv", exclude_players: set | None = None, seed: int = 11,
                   preselected: pd.DataFrame | None = None) -> None:
    """Stratified recent sample (2019+ seasons, so players are familiar), no model labels."""
    rng = np.random.default_rng(seed)
    pool = info[info._season_num >= 2019]
    if exclude_players:
        pool = pool[~pool.player.isin(exclude_players)]
    parts = []
    for (lg, dfn), n in (n_per_cell.items() if preselected is None else []):
        cell = pool[(pool.league == lg) & (pool.definition == dfn)]
        if dfn == "defined":  # spread across archetypes
            cell = cell.groupby("model_archetype", group_keys=False).apply(
                lambda g: g.sample(min(len(g), max(1, n // 6)), random_state=int(rng.integers(1e6))))
        else:
            cell = cell.sample(min(n, len(cell)), random_state=int(rng.integers(1e6)))
        parts.append(cell)
    s = pd.concat(parts).sample(frac=1, random_state=3).reset_index(drop=True) if parts else preselected.sample(
        frac=1, random_state=3).reset_index(drop=True)
    s.insert(0, "sample_id", [f"S{i + 1:03d}" for i in range(len(s))])
    key = s[["sample_id", "league", "_season_num", "_athlete_id", "player", "definition", "model_archetype", "model_subtype"]]
    key.to_csv(RESULTS / key_name, index=False)
    show = ["sample_id", "league", "season", "player", "teams", "position", "age", "games", "starts", "mpg", "pts_pg", "reb_pg",
            "ast_pg", "stl_pg", "blk_pg", "tov_pg", "fg_pct", "fg3a_pg", "fg3_pct", "usage_pct",
            "rim_shot_share_shrunk", "three_rate_shrunk"]
    path = OUT / name
    body = s[show].assign(**{c: "" for c in LABEL_COLS})
    # Format example: a player NOT in the sample, labeled the way a person would (Rudy Gobert 2024-25).
    exr = info[(info.player == "Rudy Gobert") & (info._season_num == 2025)]
    ex = (exr.assign(sample_id="EXAMPLE")[show] if len(exr) else body.iloc[[0]]).copy().assign(**{c: "" for c in LABEL_COLS})
    ex.loc[:, ["sample_id", "hand_subtype", LABEL_COLS[1], "hand_notes"]] = [
        "EXAMPLE", "Low-usage rim protector", "1", "Format example (not part of the sample) - leave as is"]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.concat([ex, body]).to_excel(xw, sheet_name="Label these", index=False)
    wb = load_workbook(path)
    ref = write_lists(wb)
    ws = wb["Label these"]
    style(ws, LABEL_COLS, {"player": 22, "teams": 12, "hand_subtype": 26, "hand_notes": 34})
    for c in ws[2]:
        c.font = Font(name="Arial", size=10, italic=True, color="808080")
    ws["A1"].comment = Comment("Fill only the yellow columns. Row 2 is an example. Model labels are hidden on purpose; "
                               "they are stored separately and compared after you finish.", "analysis")
    add_validation(ws, "hand_subtype", len(body) + 1, ref)
    wb.save(path)
    print(f"blind sample: {len(s)} rows", s.groupby(["league", "definition"]).size().to_dict())


def main() -> None:
    OUT.mkdir(exist_ok=True)
    info = player_info()
    review_workbook(info)
    # Batch 2: untouched check set. Excludes every player in batch 1 (any season); more defined
    # players and fewer limited-sample ones, since batch 1 showed limited-sample rows are hard to judge.
    b1 = pd.read_excel(OUT / "hand_labels_v1_submitted.xlsx", sheet_name="Label these")
    blind_workbook(info, {("NBA", "defined"): 18, ("NBA", "blended"): 8, ("NBA", "limited_sample"): 4,
                          ("WNBA", "defined"): 18, ("WNBA", "blended"): 8, ("WNBA", "limited_sample"): 4},
                   name="hand_label_blind_batch2.xlsx", key_name="hand_label_key_batch2.csv",
                   exclude_players=set(b1.player), seed=22)
    # Batch 3 (WNBA only, owner request): 2025-2026 seasons, mostly well-known players. One season per
    # player (the most minutes); 24 highest-minute players plus 6 rotation players with 250+ minutes.
    b2 = pd.read_excel(OUT / "hand_labels_v2_submitted.xlsx", sheet_name="Label these") \
        if (OUT / "hand_labels_v2_submitted.xlsx").exists() else pd.DataFrame({"player": []})
    seen = set(b1.player) | set(b2.player)
    w = info[(info.league == "WNBA") & info._season_num.isin([2025, 2026]) & ~info.player.isin(seen)]
    w = w.sort_values("minutes", ascending=False).drop_duplicates("player")
    top = w.head(24)
    rest = w.iloc[24:]
    bench = rest[(rest.role_tier.isin(["rotation", "deep_bench"])) & (rest.minutes >= 250)].sample(6, random_state=33)
    blind_workbook(info, {}, name="hand_label_blind_batch3_wnba.xlsx", key_name="hand_label_key_batch3.csv",
                   preselected=pd.concat([top, bench]))


if __name__ == "__main__":
    main()
