"""Step 3b: sub-types within each base archetype, role tiers, and 'undefined' players.

Sub-types: within each of the six base archetypes, K-Means (minutes-weighted) on a wider
trait set that adds scoring rate, three-point volume and accuracy, steals and blocks. For
each archetype, k_sub in 1..3 is chosen as the largest k whose split reproduces under
minutes-weighted bootstrap resampling (mean ARI >= STABLE). Names are assigned from the
sub-cluster's profile relative to its parent archetype (see SUBNAME rules).

Role tier (not style): from starts and minutes per game.
  starter          started >= 50% of games played
  rotation         otherwise, >= 30% of game minutes per game (14.4 NBA / 12 WNBA)
  deep_bench       otherwise
  limited_sample   < 250 season minutes (traits mostly reflect the prior, not the player)

Undefined, split in two:
  blended          >= 500 minutes but Gaussian-mixture membership in its own base
                   archetype < 0.5 (sits between archetypes)
  limited_sample   see above
"""
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from participation import fit_weight
from common import PROC, RESULTS, ROOT

REFERENCE = ROOT / "config" / "subtype_reference.csv"  # names + reference profiles; edit names here

SUB_FEATURES = ["usage", "ast", "pts", "fg3_rate", "fg3_pct", "stl", "blk", "orb", "drb", "rim_share", "tov"]
STABLE = 0.80
SEED = 7


def extra_z(d: pd.DataFrame) -> pd.DataFrame:
    """z-scores of shrunk traits (log for rates, logit for shares): league-season for most traits,
    league-season-position for shooting traits (POS_RELATIVE)."""
    x = pd.DataFrame(index=d.index)
    for f in ["usage", "ast", "pts", "stl", "blk", "orb", "drb", "tov"]:
        x[f] = np.log(d[f"{f}_eb"].clip(lower=1e-6))
    for f, col in [("fg3_rate", "fg3_rate_eb"), ("fg3_pct", "fg3_pct_eb"), ("rim_share", "rim_share_eb")]:
        p = d[col].clip(1e-4, 1 - 1e-4)
        x[f] = np.log(p / (1 - p))
    z = x.groupby([d.league, d.season]).transform(lambda c: (c - c.mean()) / c.std(ddof=0))
    # Shooting traits are compared with same-position peers (a center taking 30% threes is a
    # stretch big even when the league average is 42%). Groups under 15 players use the league-season.
    pos = d.position.fillna("U")
    zp = x[POS_RELATIVE].groupby([d.league, d.season, pos]).transform(lambda c: (c - c.mean()) / c.std(ddof=0))
    n = x.groupby([d.league, d.season, pos]).usage.transform("size")
    ok = (n >= 15) & (pos != "U")
    z.loc[ok, POS_RELATIVE] = zp.loc[ok, POS_RELATIVE]
    return z


POS_RELATIVE = ["fg3_rate", "fg3_pct", "rim_share"]


def stable_k(X: np.ndarray, w: np.ndarray) -> tuple[int, dict]:
    rng = np.random.default_rng(SEED)
    info, best = {}, 1
    for k in (2, 3):
        ref = KMeans(k, n_init=10, random_state=SEED).fit(X, sample_weight=w).predict(X)
        aris = []
        for b in range(10):
            idx = rng.choice(len(X), len(X), p=w / w.sum())
            aris.append(adjusted_rand_score(ref, KMeans(k, n_init=4, random_state=b).fit(X[idx]).predict(X)))
        info[k] = float(np.mean(aris))
        if info[k] >= STABLE:
            best = k
    return best, info


def name_sub(parent: str, r: pd.Series) -> str:
    """r = sub-cluster profile minus parent profile (z units)."""
    big = parent in ("Rim-running big", "Hub big / point forward")
    if big and r.fg3_rate > 0.6:
        return "Stretch / skilled big"
    if big and r.blk > 0.3 and r.usage < 0:
        return "Defensive anchor big"
    if big and (r.usage > 0.3 or r.ast > 0.3):
        return "Offensive hub big" if parent.startswith("Hub") else "Finishing big"
    if r.pts > 0.3 and r.ast < 0 and r.usage > 0.2:
        return "Microwave scorer"
    if r.ast > 0.4:
        return "Playmaker"
    if (r.stl > 0.3 or r.blk > 0.3) and r.usage < 0:
        return "Defensive specialist"
    if r.fg3_rate > 0.4 or r.fg3_pct > 0.4:
        return "Floor spacer"
    if r.orb > 0.3 or r.rim_share > 0.3:
        return "Slasher / crasher"
    return "Generalist"


def role_tier(d: pd.DataFrame) -> pd.Series:
    game_min = np.where(d.league == "NBA", 48.0, 40.0)
    mpg = d.minutes / d.games
    tier = np.select([d.minutes < 250, d.starts >= 0.5 * d.games, mpg >= 0.3 * game_min],
                     ["limited_sample", "starter", "rotation"], default="deep_bench")
    return pd.Series(tier, index=d.index)


GUARD_FAMILIES = ["On-ball creator", "Secondary guard", "Spot-up 3-and-D wing", "Connector forward"]
BIG_FAMILIES = ["Rim-running big", "Hub big / point forward"]


def position_adjust(A: pd.DataFrame) -> pd.DataFrame:
    """Make labels consistent with listed position (affects ~0.6% of 500+ minute seasons).
      C in a guard/wing archetype       -> subtype 'Stretch big (spot-up)' (e.g. B. Lopez 2019)
      G in a big archetype:
        Playmaking hub                  -> 'Jumbo playmaker' (e.g. B. Simmons, J. Giddey)
        otherwise                       -> re-assigned to the most probable guard/wing archetype,
                                           and to that archetype's nearest saved sub-type profile
    The style-only label is kept in model_archetype / model_subtype."""
    A["model_archetype"], A["model_subtype"] = A.archetype, A.subtype
    A["position_adjusted"] = False
    ref = pd.read_csv(REFERENCE) if REFERENCE.exists() else None
    c_in_wing = (A.position == "C") & A.archetype.isin(["On-ball creator", "Secondary guard", "Spot-up 3-and-D wing"])
    A.loc[c_in_wing, "subtype"] = "Stretch big (spot-up)"
    g_in_big = (A.position == "G") & A.archetype.isin(BIG_FAMILIES)
    jumbo = g_in_big & A.subtype.isin(["Playmaking hub", "Point forward / scoring hub"])
    A.loc[jumbo, "subtype"] = "Jumbo playmaker"
    move = g_in_big & ~jumbo
    if move.any():
        probs = A.loc[move, [f"prob_{f}" for f in GUARD_FAMILIES]].to_numpy()
        new_parent = np.array(GUARD_FAMILIES)[probs.argmax(1)]
        A.loc[move, "archetype"] = new_parent
        if ref is not None:
            subs = []
            for (_, row), par in zip(A.loc[move].iterrows(), new_parent):
                rr = ref[ref.archetype == par]
                dist = ((rr[SUB_FEATURES].to_numpy() - row[SUB_FEATURES].to_numpy(dtype=float)) ** 2).sum(1)
                subs.append(rr.subtype.iat[int(dist.argmin())])
            A.loc[move, "subtype"] = subs
    A.loc[c_in_wing | g_in_big, "position_adjusted"] = True
    return A


ALPHA_Z = 1.3  # usage and scoring both >= 1.3 SD above the league-season average


def alpha_override(A: pd.DataFrame) -> pd.DataFrame:
    """Owner's definition: an alpha scorer is a team's top scorer who also creates, whatever the
    style sub-type. Starters with league-season usage z and scoring-rate z both >= ALPHA_Z become
    'Alpha scorer' (rim-running bigs excluded); the style sub-type is kept in style_subtype."""
    A["style_subtype"] = A.subtype
    starter = A.starts >= 0.5 * A.games
    alpha = starter & (A.usage >= ALPHA_Z) & (A.pts >= ALPHA_Z) & (A.archetype != "Rim-running big")
    A.loc[alpha, "subtype"] = "Alpha scorer"
    A["alpha_scorer"] = alpha | (A.style_subtype == "Alpha scorer")
    return A


def main() -> None:
    A = pd.read_parquet(PROC / "archetypes.parquet")
    sh = pd.read_parquet(PROC / "player_season_shrunk.parquet")
    sh = sh[sh.minutes > 0].reset_index(drop=True)
    Z = extra_z(sh)
    Z[["league", "season", "athlete_id"]] = sh[["league", "season", "athlete_id"]]
    meta = pd.read_parquet(PROC / "player_season.parquet", columns=["league", "season", "season_type", "athlete_id", "games", "starts"])
    meta = meta[meta.season_type == "regular"].drop(columns="season_type")
    A = A.merge(Z, on=["league", "season", "athlete_id"], suffixes=("", "_x")).merge(meta, on=["league", "season", "athlete_id"])
    A = A[[c for c in A.columns if not c.endswith("_x")]]
    sub_rows, A["subtype"] = [], None
    for parent, g in A.groupby("archetype"):
        # Position is added to the sub-type step as well (same scaling as the base model), so
        # mixed groups such as hub bigs + scoring forwards can split by size. Profiles and names
        # use SUB_FEATURES only.
        pos = g.position.map({"G": -1.0, "F": 0.0, "C": 1.0}).fillna(0.0).to_numpy()
        Xs = g[SUB_FEATURES].to_numpy()
        X, w = np.column_stack([Xs, pos * np.sqrt(len(SUB_FEATURES)) / 2]), fit_weight(g)
        k, info = stable_k(X, w)
        # Lock k to the reviewed reference so small data changes (e.g. the participation filter)
        # cannot flip a borderline stability call and drop the reference names; ARIs still reported.
        n_ref = int((pd.read_csv(REFERENCE).archetype == parent).sum()) if REFERENCE.exists() else 0
        if n_ref >= 2:
            k = n_ref
        if k == 1:
            A.loc[g.index, "subtype"] = parent
            sub_rows.append({"archetype": parent, "k_sub": 1, **{f"ari_k{kk}": v for kk, v in info.items()}})
            continue
        lab = KMeans(k, n_init=20, random_state=SEED).fit(X, sample_weight=w).predict(X)
        pprof = pd.Series(np.average(Xs, axis=0, weights=w), index=SUB_FEATURES)
        cents = np.array([np.average(Xs[lab == c], axis=0, weights=w[lab == c]) for c in range(k)])
        ref = pd.read_csv(REFERENCE) if REFERENCE.exists() else pd.DataFrame(columns=["archetype"])
        ref = ref[ref.archetype == parent]
        names = {}
        if len(ref) == k:
            # Match each new sub-cluster to the closest saved reference profile (one-to-one).
            cost = ((cents[:, None, :] - ref[SUB_FEATURES].to_numpy()[None, :, :]) ** 2).sum(-1)
            r_idx, c_idx = linear_sum_assignment(cost)
            names = {int(r): ref.subtype.iat[int(c)] for r, c in zip(r_idx, c_idx)}
        for c in range(k):
            m = lab == c
            prof = pd.Series(cents[c], index=SUB_FEATURES) - pprof
            if c not in names:  # no reference (or a different number of sub-types): rule-based draft name
                nm = name_sub(parent, prof)
                names[c] = nm if nm not in names.values() else f"{nm} ({c + 1})"
            sub_rows.append({"archetype": parent, "k_sub": k, "subtype": names[c],
                             "minutes_share_of_parent": w[m].sum() / w.sum(),
                             **{f"ari_k{kk}": v for kk, v in info.items()},
                             **{f"d_{f}": round(v, 2) for f, v in prof.items()}})
        A.loc[g.index, "subtype"] = [names[c] for c in lab]
        print(f"{parent}: k_sub={k} stability={info}", file=sys.stderr)
    A = position_adjust(A)
    A = alpha_override(A)
    A["role_tier"] = role_tier(A)
    own = A.own_membership  # soft (GMM) membership in its own base archetype
    A["definition"] = np.select([A.role_tier == "limited_sample", (A.minutes >= 500) & (own < 0.5)],
                                ["limited_sample", "blended"], default="defined")
    A.to_parquet(PROC / "archetypes_detailed.parquet", index=False)
    pd.DataFrame(sub_rows).to_csv(RESULTS / "step3_subtypes.csv", index=False)


if __name__ == "__main__":
    main()
