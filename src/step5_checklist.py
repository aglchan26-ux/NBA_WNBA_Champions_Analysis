"""Step 5b: does the owner's championship checklist separate champions from other contenders?
Association only. Contenders = top-4 regular-season net rating in their league-season.
WNBA 2026 is excluded from champion comparisons: the Finals were 3-3 in the source data
(Game 7 not yet in the release)."""
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import PROC, RESULTS

ITEMS = ["C1_primary_scorer", "C2_secondary_scorer", "C3_spacing_big", "C4_perimeter_d"]


def main() -> None:
    T = pd.read_parquet(PROC / "team_seasons.parquet")
    T = T.copy()  # WNBA 2026 champion comes from a documented owner override (Game 7 not in the ESPN release)
    T["net_rank"] = T.groupby(["league", "season"]).net_rating.rank(ascending=False)
    T["group"] = np.select([T.champion, T.net_rank <= 4, T.made_playoffs], ["champion", "other contender (top-4 net)", "other playoff team"],
                           default="non-playoff")
    T["era"] = pd.cut(T.season, [2003, 2010, 2016, 2021, 2026], labels=["2004-10", "2011-16", "2017-21", "2022-26"])
    rows = []
    for (lg, grp), g in T.groupby(["league", "group"]):
        r = {"league": lg, "group": grp, "teams": len(g)}
        for c in ITEMS + ["checklist_score", "n_high_usage", "net_rating", "team_three_diet", "team_rim_diet", "C4_perimeter_d_share",
                          "rim_protector_share", "shooter_share"]:
            r[c] = g[c].mean()
        r["all_four"] = (g.checklist_score == 4).mean()
        r["two_plus_high_usage"] = (g.n_high_usage >= 2).mean()
        r["three_plus_high_usage"] = (g.n_high_usage >= 3).mean()
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(RESULTS / "step5_checklist_by_group.csv", index=False)
    # within-contender comparison: does checklist add to net rating in picking the champion?
    C = T[T.net_rank <= 4].copy()
    C["champ"] = C.champion.astype(int)
    out = []
    for lg, g in C.groupby("league"):
        for f in ["champ ~ net_rating", "champ ~ net_rating + checklist_score", "champ ~ net_rating + n_high_usage",
                  "champ ~ net_rating + C3_spacing_big_share + C4_perimeter_d_share"]:
            try:
                m = smf.logit(f, data=g).fit(disp=0)
                for v in m.params.index[1:]:
                    lo, hi = m.conf_int().loc[v]
                    out.append({"league": lg, "model": f, "term": v, "odds_ratio": np.exp(m.params[v]), "lo95": np.exp(lo),
                                "hi95": np.exp(hi), "n_contenders": len(g), "n_champions": int(g.champ.sum()),
                                "champions_in_top4": int(g.champ.sum())})
            except Exception as e:  # small samples can separate perfectly
                out.append({"league": lg, "model": f, "term": "fit failed", "odds_ratio": np.nan, "note": str(e)[:80]})
    O = pd.DataFrame(out)
    O.to_csv(RESULTS / "step5_checklist_logit.csv", index=False)
    champs_total = T.groupby("league").champion.sum()
    print(S.round(2).to_string(index=False), file=sys.stderr)
    print(O.round(2).to_string(index=False), file=sys.stderr)
    print("champions that were top-4 by net rating:", (C.groupby("league").champ.sum() / champs_total).round(2).to_dict(), file=sys.stderr)
    E = T[T.champion].groupby(["league", "era"], observed=True)[ITEMS + ["n_high_usage", "team_three_diet", "team_rim_diet",
                                                                     "C3_spacing_big_share", "C4_perimeter_d_share"]].mean()
    E.round(2).to_csv(RESULTS / "step5_champions_by_era.csv")
    print(E.round(2).to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
