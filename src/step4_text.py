"""Step 4 text layer: player-season documents built from ESPN play-by-play descriptions.

No scraping needed: every play already has a text description ("Pullup Jump Shot", "Cutting Dunk Shot",
"Bad Pass Turnover", "X blocks Y's shot", "(X assists)"). Each player-season becomes a document of the plays
credited to that player:
  * as the actor (athlete_id_1): the play type, words joined into one token, plus outcome for shots
    (e.g. pullup_jump_shot__three__made, driving_layup_shot__miss, defensive_rebound, bad_pass_turnover)
  * as the second player: assist__three / assist__rim / assist__mid (what the pass set up), block, steal
Seasons used: NBA 2017-18 onward and WNBA 2019 onward, where ESPN's play text names shot types (earlier text is
mostly "two point shot"). Player-seasons: regular season, counted by the participation rule, >= 500 minutes.

Course concepts:
  text preprocessing   lower-casing, punctuation stripping, play type -> token, outcome tags, stop-events removed
  TF-IDF               sublinear term frequency x inverse document frequency, L2-normalized
  embeddings           LSA (truncated SVD of TF-IDF, 50 dims): dense vectors that place related play types
                       together (offsets TF-IDF's blindness to related tokens). A pretrained sentence-embedding
                       model needs a download host that is blocked here; LSA needs no download.
  topic modeling       NMF on TF-IDF, k = 8 topics, top tokens per topic
  text clustering      K-Means (k = 6) on LSA embeddings, compared with the stat archetypes (ARI, NMI)
  zero-shot labeling   each archetype described in words BEFORE seeing any result (DESCRIPTIONS below);
                       a player's label = the description with the highest cosine similarity. No training
                       labels used. Evaluated against the stat archetypes and, separately, the blind hand labels.
Outputs: results/step4_*.csv, data/processed/text_embeddings.parquet
"""
import re
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize

from common import PROC, RAW, RESULTS

ESPN = RAW / "espn"
YEARS = {"NBA": range(2018, 2027), "WNBA": range(2019, 2027)}
SKIP = re.compile(r"substitution|timeout|end period|end game|end of|start|jumpball|jump ball|instant replay|"
                  r"coach|delay|ejection|official|period|game", re.I)
SEED = 3

# Zero-shot descriptions: written from basketball knowledge before any text result was looked at.
# Keywords are matched against play tokens (substring match).
DESCRIPTIONS = {
    "On-ball creator": "pullup step_back fade_away driving floating assist__three assist__rim free_throw",
    "Secondary guard": "pullup jump_shot__three assist__mid steal bad_pass running",
    "Spot-up 3-and-D wing": "jump_shot__three steal block",
    "Connector forward": "defensive_rebound cutting running jump_shot block",
    "Hub big / point forward": "assist__three assist__rim assist__mid hook post offensive_rebound defensive_rebound",
    "Rim-running big": "dunk alley_oop putback tip offensive_rebound block cutting layup",
}
# position cross-check: a zero-shot label must fit the player's listed position
COMPAT = {"On-ball creator": {"G", "F"}, "Secondary guard": {"G"}, "Spot-up 3-and-D wing": {"G", "F"},
          "Connector forward": {"F"}, "Hub big / point forward": {"F", "C"}, "Rim-running big": {"F", "C"}}


def tokens_for_season(lg: str, y: int) -> pd.DataFrame:
    p = pd.read_parquet(ESPN / f"{lg.lower()}_pbp_{y}.parquet",
                        columns=["season_type", "type_text", "text", "athlete_id_1", "athlete_id_2", "shooting_play",
                                 "scoring_play"])
    p = p[(p.season_type == 2) & p.athlete_id_1.notna()]
    p = p[~p.type_text.fillna("").str.contains(SKIP)]
    tt = p.type_text.fillna("").str.lower().str.replace(r"[^a-z0-9 ]", " ", regex=True).str.split().str.join("_")
    shot = p.shooting_play.astype(str) == "True"
    ft = tt.str.startswith("free_throw")
    made = (p.scoring_play.astype(str) == "True").to_numpy()
    txt = p.text.fillna("").str.lower()
    three = txt.str.contains("three point|3-pt|3pt")   # works in every season's files
    # one token per shot TYPE (made and missed together); a make counts twice, a miss once, so shooters who take
    # and make a shot weigh more than players who take it and miss
    tok = np.where(ft, "free_throw", np.where(shot, tt + np.where(three, "__three", ""), tt))
    tok = np.where(pd.Series(tok).str.contains("turnover|traveling|palming|double_dribble"), "turnover", tok)  # one turnover token
    wt = np.where(shot.to_numpy() | ft.to_numpy(), 1 + made.astype(int), 1)
    a = pd.DataFrame({"athlete_id": p.athlete_id_1.to_numpy(), "token": tok, "w": wt})
    rim = tt.str.contains("layup|dunk|tip|alley")
    fam = np.where(three, "three", np.where(rim, "rim", "mid"))
    extra = []
    for word, token in (("assists", "assist"), ("blocks", "block"), ("steals", "steal")):
        m = txt.str.contains(word) & p.athlete_id_2.notna()
        t = (token + "__" + pd.Series(fam, index=p.index)[m]) if token == "assist" else token   # what the pass set up
        extra.append(pd.DataFrame({"athlete_id": p.athlete_id_2[m].to_numpy(), "token": t, "w": 1}))
    d = pd.concat([a] + extra)
    d = d[d.token != ""]
    c = d.groupby(["athlete_id", "token"]).w.sum().rename("n").reset_index()
    c.insert(0, "season", y)
    c.insert(0, "league", lg)
    return c


def build_counts() -> pd.DataFrame:
    return pd.concat([tokens_for_season(lg, y) for lg, ys in YEARS.items() for y in ys], ignore_index=True)


def main() -> None:
    C = build_counts()
    A = pd.read_parquet(PROC / "archetypes_detailed.parquet", columns=["league", "season", "athlete_id", "player",
                                                                       "position", "minutes", "archetype", "subtype"])
    part = pd.read_parquet(PROC / "participation.parquet", columns=["league", "season", "athlete_id", "counted"])
    docs = A.merge(part, on=["league", "season", "athlete_id"])
    docs = docs[docs.counted & (docs.minutes >= 500)]
    docs = docs.merge(C[["league", "season", "athlete_id"]].drop_duplicates(), on=["league", "season", "athlete_id"]).reset_index(drop=True)
    key = docs[["league", "season", "athlete_id"]].assign(row=np.arange(len(docs)))
    C = C.merge(key, on=["league", "season", "athlete_id"])
    vocab = pd.Index(sorted(C.token.unique()))
    C = C[C.groupby("token").row.transform("nunique") >= 20]                 # drop rare play types (min_df = 20)
    vocab = pd.Index(sorted(C.token.unique()))
    M = sparse.csr_matrix((C.n.to_numpy(float), (C.row.to_numpy(), vocab.get_indexer(C.token))), shape=(len(docs), len(vocab)))
    X = TfidfTransformer(sublinear_tf=True).fit_transform(M)                # TF-IDF, L2-normalized rows
    print(f"documents {X.shape[0]}, play-type tokens {X.shape[1]}", file=sys.stderr)

    # distinctive tokens per stat archetype (mean TF-IDF in archetype minus overall mean)
    mean_all = np.asarray(X.mean(0)).ravel()
    rows = []
    for (lg, a), g in docs.groupby(["league", "archetype"]):
        # compared with players of the SAME listed positions (position cross-check), not the whole league
        same = docs.index[(docs.league == lg) & docs.position.isin(g.position.unique()) & (docs.archetype != a)]
        diff = np.asarray(X[g.index].mean(0)).ravel() - np.asarray(X[same].mean(0)).ravel()
        # two lists: shot types, and playmaking & defense (assists by type, blocks, steals, rebounds, turnovers),
        # so common-but-defining actions like passing are not crowded out by rare shot types
        nonshot = np.array([t.startswith(("assist", "block", "steal")) or "rebound" in t or "turnover" in t for t in vocab])
        for cat, mask in (("shots", ~nonshot), ("playmaking & defense", nonshot)):
            idx = np.where(mask)[0]
            for j in idx[np.argsort(-diff[idx])][:4]:
                if diff[j] > 0:
                    rows.append({"league": lg, "archetype": a, "category": cat, "token": vocab[j], "tfidf_lift": diff[j]})
    pd.DataFrame(rows).to_csv(RESULTS / "step4_archetype_tokens.csv", index=False)

    # topic model
    nmf = NMF(n_components=8, init="nndsvda", random_state=SEED, max_iter=600).fit(X)
    W = nmf.transform(X)
    topics = []
    for k, comp in enumerate(nmf.components_):
        top = [vocab[j] for j in np.argsort(-comp)[:8]]
        topics.append({"topic": k, "top_tokens": ", ".join(top)})
    docs["topic"] = W.argmax(1)
    T = pd.DataFrame(topics)
    share = pd.crosstab([docs.league, docs.archetype], docs.topic, normalize="index").round(3)
    T.to_csv(RESULTS / "step4_topics.csv", index=False)
    share.to_csv(RESULTS / "step4_topic_by_archetype.csv")

    # embeddings (LSA) + text clustering
    svd = TruncatedSVD(50, random_state=SEED).fit(X)
    E = normalize(svd.transform(X))
    pd.concat([docs[["league", "season", "athlete_id"]], pd.DataFrame(E, columns=[f"lsa_{i}" for i in range(50)])], axis=1) \
        .to_parquet(PROC / "text_embeddings.parquet", index=False)
    comp = []
    for lg in ["NBA", "WNBA", "both"]:
        m = (docs.league == lg).to_numpy() if lg != "both" else np.ones(len(docs), bool)
        posv = docs.position.map({"G": -1.0, "F": 0.0, "C": 1.0}).fillna(0).to_numpy()[:, None] * 0.5
        lab = KMeans(6, n_init=20, random_state=SEED).fit_predict(np.c_[E, posv][m])   # text + position
        if lg == "both":
            docs["text_cluster"] = lab
        comp.append({"league": lg, "method": "text clustering (K-Means on LSA, k=6)", "n": int(m.sum()),
                     "ari_vs_stat_archetype": adjusted_rand_score(docs.archetype[m], lab),
                     "nmi_vs_stat_archetype": normalized_mutual_info_score(docs.archetype[m], lab)})
    # name text clusters by their most over-represented tokens
    names = {}
    for c, g in docs.groupby("text_cluster"):
        lift = np.asarray(X[g.index].mean(0)).ravel() - mean_all
        names[c] = " / ".join(vocab[j] for j in np.argsort(-lift)[:3])
    docs["text_cluster_name"] = docs.text_cluster.map(names)
    ct = pd.crosstab(docs.text_cluster_name, docs.archetype)
    ct.to_csv(RESULTS / "step4_text_cluster_vs_archetype.csv")

    # zero-shot labels from the written descriptions
    Q = np.zeros((len(DESCRIPTIONS), len(vocab)))
    for i, desc in enumerate(DESCRIPTIONS.values()):
        for kw in desc.split():
            Q[i] += np.array([kw in t for t in vocab], float)
    Q = normalize(Q * np.log1p(1 / (mean_all + 1e-3)))     # rarer play types weigh more, as in IDF
    S = np.asarray((X @ Q.T).todense()) if sparse.issparse(X @ Q.T) else np.asarray(X @ Q.T)
    ok = np.array([[docs.position.iat[i] in COMPAT[a] if docs.position.iat[i] in ("G", "F", "C") else True
                    for a in DESCRIPTIONS] for i in range(len(docs))])
    S = np.where(ok, S, -np.inf)
    docs["zero_shot"] = np.array(list(DESCRIPTIONS))[S.argmax(1)]
    for lg in ["NBA", "WNBA", "both"]:
        m = (docs.league == lg) if lg != "both" else pd.Series(True, index=docs.index)
        comp.append({"league": lg, "method": "zero-shot from written descriptions", "n": int(m.sum()),
                     "ari_vs_stat_archetype": adjusted_rand_score(docs.archetype[m], docs.zero_shot[m]),
                     "nmi_vs_stat_archetype": normalized_mutual_info_score(docs.archetype[m], docs.zero_shot[m]),
                     "accuracy_vs_stat_archetype": float((docs.archetype[m] == docs.zero_shot[m]).mean())})
    # hand labels (ground truth kept separate: only read here, never used to build anything above)
    hand = pd.concat([pd.read_csv(RESULTS / f"step3_hand_label_comparison_{v}.csv") for v in ("v1", "v2", "v3")])
    hand = hand.rename(columns={"season_num": "season"})[["league", "season", "player", "hand_archetype"]].dropna()
    h = docs.merge(hand, on=["league", "season", "player"])
    for nm, col in (("zero-shot from written descriptions", "zero_shot"), ("stat archetype (Step 3)", "archetype")):
        comp.append({"league": "both", "method": f"{nm}: agreement with hand labels", "n": len(h),
                     "accuracy_vs_hand_labels": float((h[col] == h.hand_archetype).mean())})
    R = pd.DataFrame(comp)
    R.to_csv(RESULTS / "step4_text_vs_archetypes.csv", index=False)
    docs[["league", "season", "athlete_id", "player", "archetype", "topic", "text_cluster_name", "zero_shot"]] \
        .to_parquet(PROC / "text_labels.parquet", index=False)
    print(T.to_string(index=False), file=sys.stderr)
    print(R.round(3).to_string(index=False), file=sys.stderr)
    print(ct.to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
