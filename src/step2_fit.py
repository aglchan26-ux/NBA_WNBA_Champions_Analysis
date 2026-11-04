"""Step 2a: fit empirical-Bayes priors per league x season x position and shrink every player.

Outputs
  results/step2_priors.csv                 prior mean and weight k per group and feature
  data/processed/player_season_shrunk.parquet  raw, per-36 (rate stats) and shrunk values
"""
import sys

import numpy as np
import pandas as pd

from common import PROC, RESULTS
from features import PROP, RATE, ZONES, load
from shrinkage import (Prior, fit_beta_binomial, fit_dirichlet_multinomial, fit_gamma_poisson,
                       shrink, shrink_mix, within_player_dispersion)

MIN_GROUP = 20  # groups with fewer players borrow the league x position prior pooled over seasons
GROUP = ["league", "season", "position"]


def dispersion_table() -> dict:
    """phi per (league, rate feature) from regular-season player-games."""
    pg = pd.read_parquet(PROC / "player_game.parquet")
    pg = pg[pg.season_type == "regular"]
    out = {}
    for lg, g in pg.groupby("league"):
        pid = (g.athlete_id.astype(str) + "_" + g.season.astype(str)).to_numpy()
        for name, (expr, expo, _) in RATE.items():
            out[(lg, name)] = within_player_dispersion(g.eval(expr).to_numpy(), g[expo].to_numpy(), pid)
    return out


PHI: dict = {}


def fit_group(d: pd.DataFrame) -> dict:
    out = {}
    lg = d.league.iat[0]
    for name, (_, expo, _) in RATE.items():
        phi = PHI.get((lg, name), 1.0)
        pr = fit_gamma_poisson(d[f"{name}_num"] / phi, d[expo] / phi)
        # back to original units: (y + k*phi*m) / (e + k*phi)
        out[name] = Prior(mean=pr.mean, k=pr.k * phi, n_players=pr.n_players, at_bound=pr.at_bound)
    for name, (num, den, _) in PROP.items():
        out[name] = fit_beta_binomial(d[num], d[den])
    p, pr = fit_dirichlet_multinomial(d[[f"{z}_fga" for z in ZONES]].to_numpy())
    out["zone_mix"] = (p, pr)
    return out


def apply(d: pd.DataFrame, pri: dict) -> pd.DataFrame:
    d = d.copy()
    for name, (_, expo, _) in RATE.items():
        d[f"{name}_raw"] = np.where(d[expo] > 0, d[f"{name}_num"] / d[expo].where(d[expo] > 0), np.nan)
        d[f"{name}_per36"] = d[f"{name}_num"] / d.minutes * 36
        d[f"{name}_eb"] = shrink(d[f"{name}_num"], d[expo], pri[name])
    for name, (num, den, _) in PROP.items():
        d[f"{name}_raw"] = d[num] / d[den].where(d[den] > 0)
        d[f"{name}_eb"] = shrink(d[num], d[den], pri[name])
    p, pr = pri["zone_mix"]
    cnt = d[[f"{z}_fga" for z in ZONES]].to_numpy()
    mix = shrink_mix(cnt, p, pr)
    tot = cnt.sum(1, keepdims=True)
    for j, z in enumerate(ZONES):
        d[f"{z}_share_raw"] = np.where(tot[:, 0] > 0, cnt[:, j] / np.maximum(tot[:, 0], 1), np.nan)
        d[f"{z}_share_eb"] = mix[:, j]
    return d


def rows(key: tuple, pri: dict, source: str) -> list:
    r = []
    for name, v in pri.items():
        if name == "zone_mix":
            p, pr = v
            r.append({**dict(zip(GROUP, key)), "feature": name, "prior_mean": "|".join(f"{x:.3f}" for x in p),
                      "k": pr.k, "n_players": pr.n_players, "at_bound": pr.at_bound, "prior_source": source})
        else:
            r.append({**dict(zip(GROUP, key)), "feature": name, "prior_mean": v.mean, "k": v.k,
                      "n_players": v.n_players, "at_bound": v.at_bound, "prior_source": source})
    return r


def main() -> None:
    PHI.update(dispersion_table())
    pd.Series(PHI).rename("phi").rename_axis(["league", "feature"]).to_csv(RESULTS / "step2_dispersion.csv")
    d = load("regular")
    # Players with no position in any season use a league-season prior over all positions.
    d["position"] = d.position.fillna("U")
    pooled = {k: fit_group(g) for k, g in d[d.position != "U"].groupby(["league", "position"])}
    all_pos = {k: fit_group(g) for k, g in d.groupby(["league", "season"])}
    parts, prow = [], []
    for key, g in d.groupby(GROUP):
        if key[2] == "U":
            pri, src = all_pos[(key[0], key[1])], "league_season_all_positions"
        elif len(g) >= MIN_GROUP:
            pri, src = fit_group(g), "league_season_position"
        else:
            pri, src = pooled[(key[0], key[2])], "league_position_pooled"
        parts.append(apply(g, pri))
        prow += rows(key, pri, src)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(PROC / "player_season_shrunk.parquet", index=False)
    pri = pd.DataFrame(prow)
    pri.to_csv(RESULTS / "step2_priors.csv", index=False)
    print(f"players={len(out):,} groups={pri[GROUP].drop_duplicates().shape[0]} "
          f"pooled_fallback={int((pri.prior_source != 'league_season_position').sum() / pri.feature.nunique())} "
          f"at_bound={int(pri.at_bound.sum())}/{len(pri)}", file=sys.stderr)


if __name__ == "__main__":
    main()
