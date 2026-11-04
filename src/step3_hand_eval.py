"""Compare blind hand labels with the model (evaluation only; hand labels never enter a model).

Input: labeling/hand_labels_v1_submitted.xlsx (project owner's blind labels).
Matching is by league + season + player name, so regenerating the blind sheet does not matter.
Confidence (owner's guidance, 2026-11-04): keep any score the owner entered (typed on a
1 = most sure .. 3 = least sure scale, flipped here so that 3 = most sure); otherwise
  NBA: 3 if starter else 2 | WNBA 2024+: 3 if star (starter, top-quartile usage in its
  league-season) else 2 | WNBA before 2024: 1.5. "Can't tell" rows get no score and are excluded.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score

from common import PROC, RESULTS, ROOT, norm_name

SUB2ARCH = {
    "Alpha scorer": "On-ball creator", "Shot-making scorer": "On-ball creator", "Floor general": "On-ball creator",
    "Microwave scoring guard": "Secondary guard", "Defensive / connector guard": "Secondary guard",
    "Movement shooter": "Spot-up 3-and-D wing", "3-and-D wing": "Spot-up 3-and-D wing", "Catch-and-shoot wing": "Spot-up 3-and-D wing",
    "Low-usage glue defender": "Spot-up 3-and-D wing", "Scoring stretch forward": "Connector forward",
    "Perimeter stopper forward": "Connector forward", "Rim-defending big forward": "Connector forward",
    "Playmaking hub": "Hub big / point forward", "Stretch big": "Hub big / point forward",
    "Point forward / scoring hub": "Hub big / point forward", "Stretch / skilled big": "Hub big / point forward",
    "Mobile defensive big": "Rim-running big", "Interior hub big": "Hub big / point forward",
    "Rebounding / rim-running forward": "Connector forward", "Stretch big (spot-up)": "Spot-up 3-and-D wing",
    "Defensive anchor big": "Hub big / point forward", "Jumbo playmaker": "Hub big / point forward",
    "Rim-running finisher": "Rim-running big", "Low-usage rim protector": "Rim-running big",
}


def season_num(league: str, label) -> int:
    s = str(label)
    return int(s[:4]) + 1 if league == "NBA" and "-" in s else int(float(s))


# Older dropdown names -> current sub-type names (batch 2 sheet still listed some old names).
SUB_ALIAS = {"Stretch big": "Stretch / skilled big", "Playmaking hub": "Interior hub big",
             "Point forward / scoring hub": "Interior hub big", "Defensive anchor big": "Interior hub big",
             "Rim-defending big forward": "Rebounding / rim-running forward", "Finishing big": "Rim-running finisher",
             "3-and-D wing": "Catch-and-shoot wing"}


def main(batch: str = "v1") -> None:
    h = pd.read_excel(ROOT / "labeling" / f"hand_labels_{batch}_submitted.xlsx", sheet_name="Label these")
    h["hand_subtype"] = h.hand_subtype.replace(SUB_ALIAS)
    h = h[h.sample_id.astype(str).str.startswith("S")].copy()
    h["season_num"] = [season_num(lg, s) for lg, s in zip(h.league, h.season)]
    h["key"] = h.player.map(norm_name)
    A = pd.read_parquet(PROC / "archetypes_detailed.parquet")
    A["key"] = A.player.map(norm_name)
    m = h.merge(A, left_on=["league", "season_num", "key"], right_on=["league", "season", "key"],
                how="left", suffixes=("", "_m"))
    if m.archetype.isna().any():
        print("unmatched:", m[m.archetype.isna()].player.tolist(), file=sys.stderr)
    # confidence fill
    us = A.groupby(["league", "season"]).usage.quantile(.75).rename("u75")
    m = m.join(us, on=["league", "season_num"])
    star = (m.role_tier == "starter") & (m.usage >= m.u75)
    rule = np.select([m.league == "NBA", m.season_num >= 2024], [np.where(m.role_tier == "starter", 3, 2),
                     np.where(star, 3, 2)], default=1.5)
    conf_col = [c for c in m.columns if c.startswith("hand_confidence")][0]
    given = pd.to_numeric(m[conf_col], errors="coerce")
    # Owner's scale in batches 1-2 was 1 = most sure, 3 = least sure (sheet said the reverse);
    # batch 3+ sheets are labeled that way too. Internally 3 = most sure, so typed values are flipped.
    # The rule-based fill (from the owner's batch-1 message) already uses 3 = most sure.
    given = 4 - given
    m["confidence"] = given.fillna(pd.Series(rule, index=m.index))
    m["confidence_source"] = np.where(given.notna(), "owner", "rule")
    cant = m.hand_subtype.isin(["Can't tell"]) | m.hand_subtype.isna()
    m.loc[cant, "confidence"] = np.nan
    m["hand_archetype"] = m.hand_subtype.map(SUB2ARCH)
    m.loc[m.hand_subtype == "Blended / tweener", "hand_archetype"] = "Blended / tweener"
    top2 = m[[c for c in m.columns if c.startswith("prob_")]].to_numpy().argsort(1)[:, -2]
    names = np.array([c[5:] for c in m.columns if c.startswith("prob_")])
    m["model_top2_archetype"] = names[top2]
    m["agree_archetype"] = m.hand_archetype == m.archetype
    m["agree_subtype"] = m.hand_subtype == m.subtype
    m["hand_in_top2"] = m.agree_archetype | (m.hand_archetype == m.model_top2_archetype)
    keep = ["sample_id", "league", "season_num", "player", "position", "minutes", "role_tier", "definition",
            "hand_subtype", "hand_archetype", "confidence", "confidence_source", "hand_notes",
            "archetype", "subtype", "model_top2_archetype", "own_membership",
            "agree_archetype", "agree_subtype", "hand_in_top2"]
    m[keep].to_csv(RESULTS / f"step3_hand_label_comparison_{batch}.csv", index=False)

    e = m[~cant & m.hand_archetype.isin(set(SUB2ARCH.values()))]
    rows = []
    for name, g in [("all", e), *[(f"league={k}", v) for k, v in e.groupby("league")],
                    *[(f"definition={k}", v) for k, v in e.groupby("definition")],
                    *[(f"confidence>={c}", e[e.confidence >= c]) for c in (2, 3)]]:
        if len(g) == 0:
            continue
        w = g.confidence
        rows.append({"group": name, "n": len(g),
                     "archetype_agreement": g.agree_archetype.mean(),
                     "archetype_agreement_conf_weighted": np.average(g.agree_archetype, weights=w),
                     "hand_in_model_top2": g.hand_in_top2.mean(),
                     "subtype_agreement": g.agree_subtype.mean(),
                     "cohen_kappa_archetype": cohen_kappa_score(g.hand_archetype, g.archetype) if g.archetype.nunique() > 1 else np.nan,
                     "ari_archetype": adjusted_rand_score(g.hand_archetype, g.archetype)})
    s = pd.DataFrame(rows)
    # chance agreement for reference: label frequencies of each side
    s.to_csv(RESULTS / f"step3_hand_label_agreement_{batch}.csv", index=False)
    print(s.round(3).to_string(index=False), file=sys.stderr)
    print(f"excluded: can't tell={int(cant.sum())}, blended/tweener={int((m.hand_subtype == 'Blended / tweener').sum())}",
          file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v1")
