"""Step 3c: descriptive layers on top of archetype / sub-type (regular season).

Each layer is shrunk with empirical Bayes (league x season x position priors, Step 2 machinery),
compared within league-season, and only produced for seasons where its data is complete.

  creation   (NBA 2017-18+, WNBA 2019+)  shares of FGA that are drives / pull-ups / off-ball finishes
             tags at z >= 1: Downhill driver, Pull-up shooter, Off-ball finisher
  defense    (NBA 2006-07+, WNBA 2013+)  rim blocks & jump-shot blocks per opp. FGA, passing-lane &
             on-ball steals per possession, shooting fouls per possession
             tags at z >= 1: Rim protector, Jump-shot blocker, Passing-lane thief, On-ball pickpocket,
             Foul-prone (shooting fouls)
  hustle     (NBA 2017-18+, NBA.com hustle stats) deflections, contested 2s / 3s, charges drawn,
             loose balls per possession; tags at z >= 1 (charges >= 1.5): Deflection disruptor,
             Interior contester, Perimeter contester, Charge taker, Loose-ball hustler
  perimeter_d_index  mean of available z-scores: passing-lane & on-ball steals, jump-shot blocks
             (NBA 2006-07+, WNBA 2013+) and deflections & contested 3s (NBA 2017-18+)
  shot diet  (all seasons) shrunk shares of shots at the rim / midrange (4 ft to the arc) / three;
             tags at z >= 1: Rim attacker, Midrange specialist, Three-point specialist
  efficiency points per true-shot attempt (TSA = FGA + 0.44 FTA), gamma-Poisson shrinkage with the
             within-player dispersion; compared with the minutes-weighted average of the same
             archetype in the same league-season. 90% interval -> Above / Below role average, or
             'Within range' when the interval includes zero.
  microwave  scorer sub-type (Microwave scoring guard, Shot-making scorer, Alpha scorer, Scoring
             stretch forward) + reserve role (rotation / deep bench) + efficiency at or above role
             average (point estimate). Scorers in reserve roles below that are flagged separately.
z-scores use players with >= 250 minutes for the league-season mean and SD; players under 250
minutes get no tags ('limited sample').
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import PROC, RAW, RESULTS, norm_name
from shrinkage import fit_beta_binomial, fit_gamma_poisson, within_player_dispersion

CREATION_FROM = {"NBA": 2018, "WNBA": 2019}
DEFENSE_FROM = {"NBA": 2007, "WNBA": 2013}
HUSTLE_FROM = {"NBA": 2018}
SCORER_SUBTYPES = {"Microwave scoring guard", "Shot-making scorer", "Alpha scorer", "Scoring stretch forward"}
MIN_TAG = 250


def eb(d: pd.DataFrame, num: str, den: str, kind: str, phi: float = 1.0) -> pd.Series:
    """Shrunk rate/share per league x season x position (league-season prior if group < 20)."""
    out = pd.Series(np.nan, index=d.index)
    for (lg, s), g in d.groupby(["league", "season"]):
        for pos, h in g.groupby(g.position.fillna("U")):
            base = h if (len(h) >= 20 and pos != "U") else g
            if kind == "rate":
                pr = fit_gamma_poisson(base[num] / phi, base[den] / phi)
                k = pr.k * phi
            else:
                pr = fit_beta_binomial(base[num], base[den])
                k = pr.k
            out[h.index] = (h[num] + k * pr.mean) / (h[den] + k)
    return out


def zscore(d: pd.DataFrame, col: str) -> pd.Series:
    ref = d[d.minutes >= MIN_TAG].groupby(["league", "season"])[col].agg(["mean", "std"])
    j = ref.reindex(pd.MultiIndex.from_arrays([d.league, d.season]))
    return pd.Series((d[col].to_numpy() - j["mean"].to_numpy()) / j["std"].to_numpy(), index=d.index)


def tags(d: pd.DataFrame, spec: list, avail: pd.Series) -> pd.Series:
    out = []
    for i in d.index:
        if not avail[i]:
            out.append("n/a (season not covered)")
        elif d.at[i, "minutes"] < MIN_TAG:
            out.append("limited sample")
        else:
            t = [name for col, thr, name in spec if d.at[i, col] >= thr]
            out.append(" + ".join(t) if t else "none stand out")
    return pd.Series(out, index=d.index)


def hustle_table() -> pd.DataFrame:
    h = pd.read_csv(RAW / "external" / "nba_hustle.csv")
    h["key"] = h.PLAYER_NAME.map(norm_name)
    cols = {"DEFLECTIONS": "deflections", "CONTESTED_SHOTS_2PT": "contest2", "CONTESTED_SHOTS_3PT": "contest3",
            "CHARGES_DRAWN": "charges", "LOOSE_BALLS_RECOVERED": "loose_balls"}
    h = h.groupby(["key", "year"])[list(cols)].sum().rename(columns=cols).reset_index()
    return h.rename(columns={"year": "season"}).assign(league="NBA")


def main() -> None:
    A = pd.read_parquet(PROC / "archetypes_detailed.parquet")
    ps = pd.read_parquet(PROC / "player_season.parquet")
    ps = ps[ps.season_type == "regular"][["league", "season", "athlete_id", "fga", "fta", "pts", "stl", "blk", "poss_on", "opp_fga_on"]]
    ex = pd.read_parquet(PROC / "player_pbp_extras.parquet")
    ex = ex[ex.season_type == "regular"].drop(columns="season_type")
    keep = ["league", "season", "athlete_id", "player", "position", "minutes", "archetype", "subtype", "role_tier",
            "definition", "own_membership", "model_archetype", "model_subtype", "position_adjusted",
            "style_subtype", "alpha_scorer"]
    d = A[keep].merge(ps, on=["league", "season", "athlete_id"], how="left").merge(
        ex, on=["league", "season", "athlete_id"], how="left")
    for c in ex.columns.difference(["league", "season", "athlete_id"]):
        d[c] = d[c].fillna(0)
    d = d.reset_index(drop=True)

    # ---- creation
    cre = d.season >= d.league.map(CREATION_FROM)
    for c, lab in [("fga_drive", "drive"), ("fga_pullup", "pullup"), ("fga_offball", "offball")]:
        d[f"{lab}_share"] = np.nan
        d.loc[cre, f"{lab}_share"] = eb(d[cre], c, "fga", "share")
        d[f"{lab}_z"] = zscore(d, f"{lab}_share")
    d["creation_tags"] = tags(d, [("drive_z", 1, "Downhill driver"), ("pullup_z", 1, "Pull-up shooter"),
                                  ("offball_z", 1, "Off-ball finisher")], cre)
    # ---- defense detail
    dfn = d.season >= d.league.map(DEFENSE_FROM)
    for c, den in [("blk_rim", "opp_fga_on"), ("blk_jumper", "opp_fga_on"), ("stl_pass", "poss_on"),
                   ("stl_onball", "poss_on"), ("shoot_fouls", "poss_on")]:
        d[f"{c}_rate"] = np.nan
        d.loc[dfn, f"{c}_rate"] = eb(d[dfn], c, den, "rate")
        d[f"{c}_z"] = zscore(d, f"{c}_rate")
    d["defense_tags"] = tags(d, [("blk_rim_z", 1, "Rim protector"), ("blk_jumper_z", 1, "Jump-shot blocker"),
                                 ("stl_pass_z", 1, "Passing-lane thief"), ("stl_onball_z", 1, "On-ball pickpocket"),
                                 ("shoot_fouls_z", 1, "Foul-prone")], dfn)
    # ---- hustle (NBA 2017-18+)
    d["key"] = d.player.map(norm_name)
    d = d.merge(hustle_table(), on=["league", "season", "key"], how="left")
    hus = (d.league == "NBA") & (d.season >= HUSTLE_FROM["NBA"]) & d.deflections.notna()
    for c in ["deflections", "contest2", "contest3", "charges", "loose_balls"]:
        d[f"{c}_rate"] = np.nan
        d.loc[hus, f"{c}_rate"] = eb(d[hus], c, "poss_on", "rate")
        d[f"{c}_z"] = zscore(d, f"{c}_rate")
    d["hustle_tags"] = tags(d, [("deflections_z", 1, "Deflection disruptor"), ("contest2_z", 1, "Interior contester"),
                                ("contest3_z", 1, "Perimeter contester"), ("charges_z", 1.5, "Charge taker"),
                                ("loose_balls_z", 1, "Loose-ball hustler")], hus)
    # ---- shot diet (all seasons): shrunk zone shares vs league-season (Dirichlet-multinomial, Step 2)
    shz = pd.read_parquet(PROC / "player_season_shrunk.parquet",
                          columns=["league", "season", "athlete_id", "rim_share_eb", "short_mid_share_eb",
                                   "long_mid_share_eb", "corner3_share_eb", "above3_share_eb"])
    d = d.merge(shz, on=["league", "season", "athlete_id"], how="left")
    d["rim_diet"] = d.rim_share_eb
    d["mid_diet"] = d.short_mid_share_eb + d.long_mid_share_eb
    d["three_diet"] = d.corner3_share_eb + d.above3_share_eb
    for c in ["rim_diet", "mid_diet", "three_diet"]:
        d[f"{c}_z"] = zscore(d, c)
    all_seasons = pd.Series(True, index=d.index)
    d["shot_diet_tags"] = tags(d, [("rim_diet_z", 1, "Rim attacker"), ("mid_diet_z", 1, "Midrange specialist"),
                                   ("three_diet_z", 1, "Three-point specialist")], all_seasons)
    d["main_shot_zone"] = d[["rim_diet", "mid_diet", "three_diet"]].idxmax(axis=1).str.replace("_diet", "")
    # three-point diet vs same-position peers (for "spacing big"); reference = 250+ minute players
    pos = d.position.fillna("U")
    ref = d[d.minutes >= MIN_TAG].groupby([d.league, d.season, pos]).three_diet.agg(["mean", "std"])
    j = ref.reindex(pd.MultiIndex.from_arrays([d.league, d.season, pos]))
    d["three_diet_pos_z"] = (d.three_diet.to_numpy() - j["mean"].to_numpy()) / j["std"].to_numpy()
    # ---- defensive activity index (perimeter): mean of the available event z-scores
    dz = d[["stl_pass_z", "stl_onball_z", "blk_jumper_z", "deflections_z", "contest3_z"]]
    d["perimeter_d_index"] = dz.mean(axis=1, skipna=True)
    # ---- efficiency vs role
    pg = pd.read_parquet(PROC / "player_game.parquet", columns=["league", "season", "season_type", "athlete_id", "pts", "fga", "fta"])
    pg = pg[pg.season_type == "regular"]
    phi = {lg: within_player_dispersion(g.pts.to_numpy(), (g.fga + 0.44 * g.fta).to_numpy(),
                                        (g.athlete_id.astype(str) + g.season.astype(str)).to_numpy())
           for lg, g in pg.groupby("league")}
    d["tsa"] = d.fga + 0.44 * d.fta
    a_post, b_post = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for (lg, s, pos), g in d.groupby(["league", "season", d.position.fillna("U")]):
        base = g if (len(g) >= 20 and pos != "U") else d[(d.league == lg) & (d.season == s)]
        f = phi[lg]
        pr = fit_gamma_poisson(base.pts / f, base.tsa / f)
        a0, b0 = pr.k * pr.mean, pr.k  # gamma prior on points per TSA, in units of y/phi
        a_post[g.index] = a0 + g.pts / f
        b_post[g.index] = b0 + g.tsa / f
    d["pts_per_tsa_eb"] = a_post / b_post
    d["pts_per_tsa_sd"] = np.sqrt(a_post) / b_post
    role_mean = d.groupby(["league", "season", "archetype"]).apply(
        lambda g: np.average(g.pts_per_tsa_eb, weights=g.minutes)).rename("role_mean")
    d = d.join(role_mean, on=["league", "season", "archetype"])
    d["eff_vs_role"] = d.pts_per_tsa_eb - d.role_mean
    zq = stats.norm.ppf(0.95)
    d["eff_lo90"], d["eff_hi90"] = d.eff_vs_role - zq * d.pts_per_tsa_sd, d.eff_vs_role + zq * d.pts_per_tsa_sd
    d["efficiency_tag"] = np.select([d.minutes < MIN_TAG, d.eff_lo90 > 0, d.eff_hi90 < 0],
                                    ["limited sample", "Above role average", "Below role average"], default="Within range")
    # ---- microwave composite
    # Efficiency must be at or above the role average on the point estimate (not merely "not
    # significantly below"): a high-volume bench scorer with sub-par efficiency is a bench
    # scorer, not a microwave (owner's example: Scoot Henderson 2025-26).
    d["bench_microwave"] = d.subtype.isin(SCORER_SUBTYPES) & d.role_tier.isin(["rotation", "deep_bench"]) & (
        d.eff_vs_role >= 0) & (d.minutes >= MIN_TAG)
    d["bench_scorer_below_avg_eff"] = d.subtype.isin(SCORER_SUBTYPES) & d.role_tier.isin(["rotation", "deep_bench"]) & (
        d.eff_vs_role < 0) & (d.minutes >= MIN_TAG)
    d.drop(columns="key").to_parquet(PROC / "player_layers.parquet", index=False)

    # ---- check: do defensive sub-types show more hustle / defensive events? (NBA 2017-18+, 500+ min)
    v = d[hus & (d.minutes >= 500)]
    chk = v.groupby("subtype")[["deflections_z", "contest2_z", "contest3_z", "charges_z", "stl_pass_z", "blk_rim_z"]].mean()
    chk["n"] = v.groupby("subtype").size()
    chk.round(2).sort_values("deflections_z", ascending=False).to_csv(RESULTS / "step3_hustle_by_subtype.csv")
    summ = {}
    for col in ["creation_tags", "shot_diet_tags", "defense_tags", "hustle_tags", "efficiency_tag"]:
        summ[col] = d[col].str.split(" \\+ ").explode().value_counts()
    pd.concat(summ, axis=1).to_csv(RESULTS / "step3_layer_tag_counts.csv")
    print(chk.round(2).sort_values("deflections_z", ascending=False).to_string(), file=sys.stderr)
    print("dispersion pts per TSA:", {k: round(v, 2) for k, v in phi.items()}, "| bench microwave:",
          int(d.bench_microwave.sum()), file=sys.stderr)


if __name__ == "__main__":
    main()
