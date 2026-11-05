"""Reputation layer: how teams are DESCRIBED (Wikipedia team-season articles) vs how they PLAYED (data).

Input: data/raw/text/wikipedia/*.json from scripts/fetch_team_text.py (run locally). If no files are there,
the script says so and exits without writing results.

Per team-season article:
  preprocessing   plain text -> lower case -> word tokens; stop words removed (scikit-learn English list)
  TF-IDF          distinctive words of each article vs all articles of that league (top 12 saved)
  offense/defense emphasis  mentions of defense terms vs offense terms per 1,000 words, and
                  lean = (defense - offense) / (defense + offense)   (-1 = all offense talk, +1 = all defense talk)
  sentiment       share of positive minus negative words (small built-in lexicon, listed below), per 1,000 words
Comparison: lean vs the team's measured identity (defense edge minus offense edge, from
step5_champion_identity / team ratings); correlation by league with a bootstrap interval.
Outputs: results/step4_reputation.csv, step4_reputation_vs_data.csv
"""
import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from common import PROC, RAW, RESULTS

DEF = {"defense", "defensive", "defend", "defended", "defending", "defender", "defenders", "stops", "stop", "blocks",
       "blocked", "steals", "rim", "protector", "perimeter", "lockdown", "stingy", "held", "holding", "forced", "turnovers"}
OFF = {"offense", "offensive", "scoring", "scored", "scorer", "points", "shooting", "shooter", "three", "threes",
       "pointers", "efficient", "efficiency", "assists", "passing", "transition", "pace", "fast"}
POS = {"win", "won", "wins", "winning", "dominant", "dominated", "best", "record", "champion", "championship", "title",
       "historic", "strong", "improved", "success", "successful", "led", "clinched", "streak", "upset"}
NEG = {"loss", "lost", "losses", "losing", "injury", "injured", "injuries", "struggled", "struggles", "worst",
       "eliminated", "swept", "suspended", "slump", "disappointing", "missed", "out", "decline", "collapsed"}


def load_texts() -> pd.DataFrame:
    d = RAW / "text" / "wikipedia"
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(d.glob("*.json"))] if d.exists() else []
    rows = [r for r in rows if not r.get("missing") and len(r.get("text", "")) > 500]
    return pd.DataFrame(rows)


def main() -> None:
    T = load_texts()
    if T.empty:
        print("No article text found in data/raw/text/wikipedia/. Run scripts/fetch_team_text.py on a computer "
              "that can reach en.wikipedia.org, then rerun this script.", file=sys.stderr)
        return
    words = T.text.str.lower().str.findall(r"[a-z]+")
    n = words.str.len().clip(lower=1)
    T["defense_per_1k"] = 1000 * words.map(lambda w: sum(x in DEF for x in w)) / n
    T["offense_per_1k"] = 1000 * words.map(lambda w: sum(x in OFF for x in w)) / n
    T["defense_lean"] = (T.defense_per_1k - T.offense_per_1k) / (T.defense_per_1k + T.offense_per_1k).replace(0, np.nan)
    T["sentiment_per_1k"] = 1000 * words.map(lambda w: sum(x in POS for x in w) - sum(x in NEG for x in w)) / n
    tops = []
    for lg, g in T.groupby("league"):
        v = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS), sublinear_tf=True, min_df=2, max_df=0.6,
                            token_pattern=r"(?u)\b[a-z][a-z]+\b")
        X = v.fit_transform(g.text.str.lower())
        vocab = np.array(v.get_feature_names_out())
        for i, idx in enumerate(g.index):
            row = X[i].toarray().ravel()
            tops.append((idx, ", ".join(vocab[np.argsort(-row)[:12]])))
    T["distinctive_words"] = pd.Series(dict(tops))
    keep = ["league", "season", "team_id", "title", "defense_per_1k", "offense_per_1k", "defense_lean", "sentiment_per_1k",
            "distinctive_words"]
    T[keep].to_csv(RESULTS / "step4_reputation.csv", index=False)

    ts = pd.read_parquet(PROC / "team_seasons.parquet", columns=["league", "season", "team_id", "ortg", "drtg"])
    g = ts.groupby(["league", "season"])
    ts["identity_def_minus_off"] = (g.drtg.transform("mean") - ts.drtg) - (ts.ortg - g.ortg.transform("mean"))
    J = T.merge(ts, on=["league", "season", "team_id"]).dropna(subset=["defense_lean"])
    rng = np.random.default_rng(5)
    out = []
    for lg, d in J.groupby("league"):
        if len(d) < 10:  # too few articles for a correlation
            continue
        r = np.corrcoef(d.defense_lean, d.identity_def_minus_off)[0, 1]
        b = [np.corrcoef(*d.sample(len(d), replace=True, random_state=int(rng.integers(1e9)))[["defense_lean", "identity_def_minus_off"]].to_numpy().T)[0, 1]
             for _ in range(1000)]
        out.append({"league": lg, "articles": len(d), "corr_reputation_vs_data": r,
                    "lo95": np.nanpercentile(b, 2.5), "hi95": np.nanpercentile(b, 97.5)})
    pd.DataFrame(out).to_csv(RESULTS / "step4_reputation_vs_data.csv", index=False)
    print(pd.DataFrame(out).round(3).to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
