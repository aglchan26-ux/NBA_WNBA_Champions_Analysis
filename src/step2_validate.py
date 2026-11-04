"""Step 2b: does shrinkage predict better than raw rates and per-36?

Test A (split-half): each league-season's regular season is split at the median game date.
  Estimates from half 1 predict half-2 counts. Players must appear in both halves.
Test B (regular -> playoffs): full regular-season estimates predict playoff counts.
  (No opponent adjustment here; every method faces the same tougher playoff defenses.)

Methods
  raw        half-1 rate per possession (or per chance / attempt)
  per36      half-1 rate per minute, scaled by half-2 minutes (counting stats only)
  eb_season  empirical-Bayes shrinkage, prior per league x season x position (Step 2a)
  eb_pooled  same, prior pooled over all seasons up to and including this one (expanding)
  ridge      L2-penalized Poisson / logistic player effects around the group mean; penalty
             chosen by leave-one-season-out on the same split (never sees its own season)
When raw has no estimate (zero exposure in half 1), it gets the group mean - this favors raw.

Score: mean deviance of half-2 (or playoff) counts per player (Poisson for rates, binomial
for percentages, multinomial for shot-zone mix). Lower is better. Reported overall and by
half-1 minutes bucket.
"""
import sys

import numpy as np
import pandas as pd

import step2_fit as S
from common import PROC, RESULTS
from features import PROP, RATE, ZONES, load

# Half-season (or playoff) minutes; same cut points for both leagues.
BUCKETS = [0, 100, 250, 500, 1e6]
BLAB = ["<100", "100-250", "250-500", "500+"]
LAMBDAS = np.logspace(-1, 3, 13)
KEYS = ["league", "season", "athlete_id"]


# ---------- data ----------
def game_zone_counts() -> pd.DataFrame:
    s = pd.read_parquet(PROC / "shots.parquet", columns=["game_id", "athlete_id", "zone", "made", "assisted"])
    s = s[s.zone.isin(ZONES)]
    a = s.assign(n=1).pivot_table(index=["game_id", "athlete_id"], columns="zone", values="n", aggfunc="sum", fill_value=0)
    m = s.assign(n=s.made.astype(int)).pivot_table(index=["game_id", "athlete_id"], columns="zone", values="n", aggfunc="sum", fill_value=0)
    a.columns = [f"{c}_fga" for c in a.columns]
    m.columns = [f"{c}_fgm" for c in m.columns]
    out = a.join(m)
    g = s.groupby(["game_id", "athlete_id"])
    out["pbp_fgm"] = g.made.sum()
    out["assisted_fgm"] = g.assisted.sum()
    return out.reset_index()


def aggregate(pg: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in pg.columns if c.endswith(("_fga", "_fgm")) or c in
           ["minutes", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "orb", "drb", "ast", "stl", "blk", "tov", "pf",
            "pts", "poss_on", "orb_chances_on", "drb_chances_on", "tm_fgm_on", "opp_fga_on", "assisted_fgm"]]
    return pg.groupby(KEYS)[num].sum().reset_index()


def split_data():
    pg = pd.read_parquet(PROC / "player_game.parquet")
    pg = pg.merge(game_zone_counts(), on=["game_id", "athlete_id"], how="left").fillna(
        {c: 0 for c in [f"{z}_{s}" for z in ZONES for s in ("fga", "fgm")] + ["pbp_fgm", "assisted_fgm"]})
    pos = pd.read_parquet(PROC / "player_season.parquet", columns=["league", "season", "season_type", "athlete_id", "position"])
    pos = pos[pos.season_type == "regular"].drop(columns="season_type")
    pos["position"] = pos.position.fillna("U")
    pg["game_date"] = pd.to_datetime(pg.game_date)
    reg = pg[pg.season_type == "regular"]
    mid = reg.groupby(["league", "season"]).game_date.transform("median")
    h1 = aggregate(reg[reg.game_date < mid]).merge(pos, on=KEYS)
    h2 = aggregate(reg[reg.game_date >= mid]).merge(pos, on=KEYS)
    full = aggregate(reg).merge(pos, on=KEYS)
    po = aggregate(pg[pg.season_type == "playoffs"]).merge(pos, on=KEYS)
    return [load(source=x) for x in (h1, h2, full, po)]


# ---------- estimators ----------
def eb_fit(train: pd.DataFrame, pooled: bool) -> pd.DataFrame:
    parts = []
    for lg, dl in train.groupby("league"):
        seasons = sorted(dl.season.unique())
        for s in seasons:
            cur = dl[dl.season == s]
            base = dl[dl.season <= s] if pooled else cur
            pos_fit = {p: S.fit_group(g) for p, g in base[base.position != "U"].groupby("position")}
            allp = S.fit_group(base)
            for p, g in cur.groupby("position"):
                pri = pos_fit[p] if (p in pos_fit and (pooled or len(g) >= S.MIN_GROUP)) else allp
                parts.append(S.apply(g, pri))
    return pd.concat(parts, ignore_index=True)


def ridge_poisson(y, e, mu, lam, iters=25):
    """argmin_b  -[y*(mu+b) - e*exp(mu+b)] + lam*b^2, vectorized Newton."""
    b = np.zeros_like(y, dtype=float)
    for _ in range(iters):
        r = e * np.exp(mu + b)
        b -= (r - y + 2 * lam * b) / (r + 2 * lam)
    return np.exp(mu + b)


def ridge_logit(k, n, mu, lam, iters=25):
    b = np.zeros_like(k, dtype=float)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(mu + b, -30, 30)))
        b -= (n * p - k + 2 * lam * b) / (n * p * (1 - p) + 2 * lam)
    return 1 / (1 + np.exp(-np.clip(mu + b, -30, 30)))


def group_mu(d, num, den, logit):
    g = d.groupby(["league", "season", "position"])
    rate = (g[num].transform("sum") / g[den].transform("sum")).clip(1e-6, 1 - 1e-6 if logit else None)
    return np.log(rate / (1 - rate)) if logit else np.log(rate)


# ---------- scoring ----------
def pois_dev(y, mu):
    mu = np.maximum(mu, 1e-9)
    return 2 * (np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0) - (y - mu))


def binom_dev(k, n, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -2 * (k * np.log(p) + (n - k) * np.log(1 - p))  # relative deviance (constant dropped)


def evaluate(train: pd.DataFrame, test: pd.DataFrame, label: str) -> pd.DataFrame:
    ebs = eb_fit(train, pooled=False).set_index(KEYS)
    ebp = eb_fit(train, pooled=True).set_index(KEYS)
    tr = train.set_index(KEYS)
    te = test.set_index(KEYS)
    idx = tr.index.intersection(te.index)
    tr, te, ebs, ebp = tr.loc[idx], te.loc[idx], ebs.loc[idx], ebp.loc[idx]
    bucket = pd.cut(tr.minutes, BUCKETS, labels=BLAB)
    rows = []

    def emit(feature, method, dev, n_eval):
        df = pd.DataFrame({"dev": np.asarray(dev), "bucket": bucket.to_numpy(), "league": tr.reset_index().league.to_numpy(),
                           "season": tr.reset_index().season.to_numpy(), "ok": n_eval > 0})
        df = df[df.ok]
        for (lg, b), g in df.groupby(["league", "bucket"], observed=True):
            rows.append({"test": label, "league": lg, "bucket": b, "feature": feature, "method": method,
                         "n_players": len(g), "mean_dev": g.dev.mean()})
        for lg, g in df.groupby("league"):
            rows.append({"test": label, "league": lg, "bucket": "all", "feature": feature, "method": method,
                         "n_players": len(g), "mean_dev": g.dev.mean()})
        return df

    trr = tr.reset_index()
    for name, (_, expo, _) in RATE.items():
        y, e2 = te[f"{name}_num"].to_numpy(), te[expo].to_numpy()
        mu_g = group_mu(trr, f"{name}_num", expo, False).to_numpy()
        raw = np.where(tr[expo] > 0, tr[f"{name}_num"] / tr[expo].where(tr[expo] > 0), np.exp(mu_g))
        per_min = np.where(tr.minutes > 0, tr[f"{name}_num"] / tr.minutes, 0)
        preds = {"raw": raw * e2, "per36": per_min * te.minutes.to_numpy(),
                 "eb_season": ebs[f"{name}_eb"].to_numpy() * e2, "eb_pooled": ebp[f"{name}_eb"].to_numpy() * e2}
        yield_ridge = {lam: ridge_poisson(tr[f"{name}_num"].to_numpy(), tr[expo].to_numpy(), mu_g, lam) for lam in LAMBDAS}
        preds["ridge"] = loso_pick(yield_ridge, lambda r: pois_dev(y, r * e2), trr) * e2
        for m, p in preds.items():
            emit(name, m, pois_dev(y, p), e2)
    for name, (num, den, _) in PROP.items():
        k, n = te[num].to_numpy(), te[den].to_numpy()
        mu_g = group_mu(trr, num, den, True).to_numpy()
        raw = np.where(tr[den] > 0, tr[num] / tr[den].where(tr[den] > 0), 1 / (1 + np.exp(-mu_g)))
        preds = {"raw": raw, "eb_season": ebs[f"{name}_eb"].to_numpy(), "eb_pooled": ebp[f"{name}_eb"].to_numpy()}
        cand = {lam: ridge_logit(tr[num].to_numpy(), tr[den].to_numpy(), mu_g, lam) for lam in LAMBDAS}
        preds["ridge"] = loso_pick(cand, lambda p: binom_dev(k, n, p), trr)
        for m, p in preds.items():
            emit(name, m, binom_dev(k, n, p), n)
    # shot-zone mix (multinomial)
    c2 = te[[f"{z}_fga" for z in ZONES]].to_numpy()
    c1 = tr[[f"{z}_fga" for z in ZONES]].to_numpy()
    tot1 = c1.sum(1, keepdims=True)
    gmix = trr.groupby(["league", "season", "position"])[[f"{z}_fga" for z in ZONES]].transform("sum").to_numpy()
    gmix = gmix / gmix.sum(1, keepdims=True)
    raw = np.where(tot1 > 0, c1 / np.maximum(tot1, 1), gmix)
    for m, p in {"raw": raw, "eb_season": ebs[[f"{z}_share_eb" for z in ZONES]].to_numpy(),
                 "eb_pooled": ebp[[f"{z}_share_eb" for z in ZONES]].to_numpy()}.items():
        emit("zone_mix", m, -2 * (c2 * np.log(np.clip(p, 1e-6, 1))).sum(1), c2.sum(1))
    return pd.DataFrame(rows)


def loso_pick(cands: dict, devfun, trr: pd.DataFrame) -> np.ndarray:
    """For each league-season, use the lambda that minimizes deviance on the OTHER seasons."""
    dev = {lam: devfun(p) for lam, p in cands.items()}
    key = trr.league.astype(str) + "_" + trr.season.astype(str)
    out = np.empty(len(trr))
    tab = pd.DataFrame({lam: pd.Series(d).groupby(key.to_numpy()).sum() for lam, d in dev.items()})
    lg_of = pd.Series(tab.index.str.split("_").str[0], index=tab.index)
    for ks in tab.index:
        other = tab[(lg_of == lg_of[ks]) & (tab.index != ks)].sum()
        best = other.idxmin()
        m = (key == ks).to_numpy()
        out[m] = cands[best][m]
    return out


def main() -> None:
    h1, h2, full, po = split_data()
    a = evaluate(h1, h2, "split_half")
    b = evaluate(full, po, "regular_to_playoffs")
    res = pd.concat([a, b], ignore_index=True)
    raw = res[res.method == "raw"].set_index(["test", "league", "bucket", "feature"]).mean_dev
    res["vs_raw_pct"] = 100 * (res.mean_dev / raw.reindex(res.set_index(["test", "league", "bucket", "feature"]).index).to_numpy() - 1)
    res.to_csv(RESULTS / "step2_validation.csv", index=False)
    print(f"rows={len(res)}", file=sys.stderr)


if __name__ == "__main__":
    main()
