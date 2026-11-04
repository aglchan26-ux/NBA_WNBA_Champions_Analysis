"""Step 3d: archetypes for parts of a season - team stints (trades) and season halves.

A segment's counts are shrunk with the same league x season x position priors as full seasons,
z-scored with the saved player-season norms, and assigned with the saved K-Means model (base
archetype) and the nearest saved sub-type profile. This lets us ask:
  - Does a player's assigned role change after a mid-season trade?
  - As a baseline, how often does it change between the two halves of a season for players who
    stayed on one team? If trades change roles more often than the baseline, the labels are
    picking up real role changes, not just noise.
Segments under MIN_SEG minutes are skipped.
"""
import pickle
import sys

import numpy as np
import pandas as pd

import step2_fit as S2
from common import PROC, RESULTS, ROOT
from features import ZONES, load
from step3_archetypes import FEATURES, model_matrix, style_matrix
from step3_subtypes import SUB_FEATURES

MIN_SEG = 300
REFERENCE = ROOT / "config" / "subtype_reference.csv"
SUM_COLS = ["minutes", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "orb", "drb", "ast", "stl", "blk", "tov", "pf",
            "pts", "poss_on", "orb_chances_on", "drb_chances_on", "tm_fgm_on", "opp_fga_on"]


def segments() -> pd.DataFrame:
    pg = pd.read_parquet(PROC / "player_game.parquet")
    pg = pg[pg.season_type == "regular"].copy()
    pg["game_date"] = pd.to_datetime(pg.game_date)
    pg = pg.sort_values("game_date")
    n_team = pg.groupby(["league", "season", "athlete_id"]).team_id.transform("nunique")
    # traded: one segment per team stint; others: first vs second half of the player's own games
    order = pg.groupby(["league", "season", "athlete_id"]).cumcount()
    ngames = pg.groupby(["league", "season", "athlete_id"]).game_id.transform("count")
    pg["seg_type"] = np.where(n_team > 1, "team_stint", "season_half")
    pg["seg"] = np.where(n_team > 1, pg.team_abbreviation.astype(str),
                         np.where(order < ngames / 2, "H1", "H2"))
    shots = pd.read_parquet(PROC / "shots.parquet", columns=["game_id", "athlete_id", "zone", "made", "assisted"])
    a = shots.assign(n=1).pivot_table(index=["game_id", "athlete_id"], columns="zone", values="n", aggfunc="sum", fill_value=0)
    m = shots.assign(n=shots.made.astype(int)).pivot_table(index=["game_id", "athlete_id"], columns="zone", values="n", aggfunc="sum", fill_value=0)
    a.columns, m.columns = [f"{c}_fga" for c in a.columns], [f"{c}_fgm" for c in m.columns]
    z = a.join(m)
    z["pbp_fgm"] = shots.groupby(["game_id", "athlete_id"]).made.sum()
    z["assisted_fgm"] = shots.groupby(["game_id", "athlete_id"]).assisted.sum()
    pg = pg.merge(z.reset_index(), on=["game_id", "athlete_id"], how="left")
    zc = [c for c in z.columns]
    pg[zc] = pg[zc].fillna(0)
    keys = ["league", "season", "athlete_id", "seg_type", "seg"]
    seg = pg.groupby(keys).agg(**{c: (c, "sum") for c in SUM_COLS + zc},
                               games=("game_id", "nunique"), first_date=("game_date", "min"),
                               player=("athlete_display_name", "last")).reset_index()
    pos = pd.read_parquet(PROC / "archetypes_detailed.parquet", columns=["league", "season", "athlete_id", "position",
                                                                         "archetype", "subtype"])
    return seg.merge(pos.rename(columns={"archetype": "season_archetype", "subtype": "season_subtype"}),
                     on=["league", "season", "athlete_id"], how="inner")


def sub_z(d: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Sub-type features for segments, z-scored with player-season norms."""
    def raw(x):
        r = pd.DataFrame(index=x.index)
        for f in ["usage", "ast", "pts", "stl", "blk", "orb", "drb", "tov"]:
            r[f] = np.log(x[f"{f}_eb"].clip(lower=1e-6))
        for f, col in [("fg3_rate", "fg3_rate_eb"), ("fg3_pct", "fg3_pct_eb"), ("rim_share", "rim_share_eb")]:
            p = x[col].clip(1e-4, 1 - 1e-4)
            r[f] = np.log(p / (1 - p))
        return r
    rr = raw(ref)
    g = rr.groupby([ref.league, ref.season])
    mu, sd = g.mean(), g.std(ddof=0)
    idx = pd.MultiIndex.from_arrays([d.league, d.season])
    return pd.DataFrame((raw(d).to_numpy() - mu.reindex(idx).to_numpy()) / sd.reindex(idx).to_numpy(),
                        index=d.index, columns=rr.columns)


def main() -> None:
    seg = segments()
    seg = seg[seg.minutes >= MIN_SEG].reset_index(drop=True)
    full = load("regular")
    full["position"] = full.position.fillna("U")
    seg_l = load(source=seg)
    seg_l["position"] = seg_l.position.fillna("U")
    parts = []
    for (lg, s, p), g in seg_l.groupby(["league", "season", "position"]):
        base = full[(full.league == lg) & (full.season == s)]
        b2 = base[base.position == p]
        pri = S2.fit_group(b2 if (len(b2) >= S2.MIN_GROUP and p != "U") else base)
        parts.append(S2.apply(g, pri))
    sh = pd.concat(parts)
    ref_sh = pd.read_parquet(PROC / "player_season_shrunk.parquet")
    norms = pd.read_parquet(PROC / "style_norms.parquet")
    Z = style_matrix(sh, norms)
    with open(PROC / "archetype_models.pkl", "rb") as fh:
        M = pickle.load(fh)
    Xs = pd.concat([sh[["position"]], Z], axis=1)
    sh["archetype"] = pd.Series(M["kmeans"].predict(model_matrix(Xs)), index=sh.index).map(M["names"])
    SZ = sub_z(sh, ref_sh[ref_sh.minutes > 0])
    ref = pd.read_csv(REFERENCE)
    subs = []
    for i in sh.index:
        rr = ref[ref.archetype == sh.at[i, "archetype"]]
        dist = ((rr[SUB_FEATURES].to_numpy() - SZ.loc[i, SUB_FEATURES].to_numpy(dtype=float)) ** 2).sum(1)
        subs.append(rr.subtype.iat[int(dist.argmin())])
    sh["subtype"] = subs
    out = sh[["league", "season", "athlete_id", "player", "position", "seg_type", "seg", "first_date", "games", "minutes",
              "season_archetype", "season_subtype", "archetype", "subtype", "usage_eb", "ast_eb", "pts_eb",
              "fg3_rate_eb", "rim_share_eb", "assisted_eb"]].sort_values(["league", "season", "athlete_id", "first_date"])
    out.to_parquet(PROC / "segments_archetypes.parquet", index=False)

    # change rates: consecutive segments of the same player-season, both >= MIN_SEG minutes
    pairs = []
    for (lg, s, a, t), g in out.groupby(["league", "season", "athlete_id", "seg_type"]):
        if len(g) < 2:
            continue
        g = g.sort_values("first_date")
        for i in range(len(g) - 1):
            x, y = g.iloc[i], g.iloc[i + 1]
            pairs.append({"league": lg, "season": s, "athlete_id": a, "player": x.player, "seg_type": t,
                          "from": x.seg, "to": y.seg, "min_from": x.minutes, "min_to": y.minutes,
                          "arch_from": x.archetype, "arch_to": y.archetype, "sub_from": x.subtype, "sub_to": y.subtype,
                          "usage_from": x.usage_eb, "usage_to": y.usage_eb, "ast_from": x.ast_eb, "ast_to": y.ast_eb,
                          "fg3_rate_from": x.fg3_rate_eb, "fg3_rate_to": y.fg3_rate_eb})
    P = pd.DataFrame(pairs)
    P["arch_changed"] = P.arch_from != P.arch_to
    P["sub_changed"] = P.sub_from != P.sub_to
    P.to_csv(RESULTS / "step3_segment_pairs.csv", index=False)
    rows = []
    for (lg, t), g in P.groupby(["league", "seg_type"]):
        n = len(g)
        for col in ["arch_changed", "sub_changed"]:
            p = g[col].mean()
            se = np.sqrt(p * (1 - p) / n)
            rows.append({"league": lg, "comparison": "after trade" if t == "team_stint" else "same team, half 1 vs 2",
                         "measure": col, "n_pairs": n, "rate": p, "lo95": max(0, p - 1.96 * se), "hi95": min(1, p + 1.96 * se)})
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "step3_role_change_rates.csv", index=False)
    print(R.round(3).to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
