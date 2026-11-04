"""Step 2 figures and checkpoint tables (reads saved outputs only)."""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIGS, PROC, RESULTS

C1, C2, C3, GRAY, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#9b9a95", "#e5e4e0", "#fcfcfb"
BUCKETS = [0, 100, 300, 700, 1500, 1e5]
LABELS = ["<100", "100-300", "300-700", "700-1500", "1500+"]


def movement_table(d: pd.DataFrame) -> pd.DataFrame:
    d = d.assign(bucket=pd.cut(d.minutes, BUCKETS, labels=LABELS))
    rows = []
    for f in ["pts", "usage", "ast", "orb", "stl", "fg3_pct", "rim_pct", "fg3_rate"]:
        sd = d.groupby("league")[f + "_eb"].transform("std")
        mv = (d[f + "_raw"] - d[f + "_eb"]).abs() / sd
        rows.append(mv.groupby([d.league, d.bucket], observed=True).median().rename(f))
    return pd.concat(rows, axis=1)


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    d = pd.read_parquet(PROC / "player_season_shrunk.parquet")
    t = movement_table(d)
    t.round(3).to_csv(RESULTS / "step2_shrinkage_movement.csv")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
    x = d[(d.league == "NBA") & (d.season == 2025) & (d.fg3a > 0)]
    ax[0].scatter(x.fg3a, x.fg3_pct_raw, s=10, c=GRAY, alpha=.6, label="Raw 3P%", linewidths=0)
    ax[0].scatter(x.fg3a, x.fg3_pct_eb, s=10, c=C1, alpha=.85, label="Shrunk 3P%", linewidths=0)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Three-point attempts (log scale)")
    ax[0].set_ylabel("Three-point %")
    ax[0].set_title("NBA 2024-25: small samples are pulled toward the prior", fontsize=10, loc="left")
    ax[0].legend(frameon=False, fontsize=8)
    m = t.xs("NBA")
    xs, w = np.arange(len(m)), .27
    for i, (f, c, lab) in enumerate([("pts", C1, "Points per possession"), ("fg3_pct", C2, "Three-point %"),
                                     ("stl", C3, "Steals per possession")]):
        ax[1].bar(xs + (i - 1) * w, m[f], w * .92, color=c, label=lab)
    ax[1].set_xticks(xs, m.index)
    ax[1].set_xlabel("Season minutes")
    ax[1].set_ylabel("Median shift, in between-player SDs")
    ax[1].set_title("How far shrinkage moves a player (NBA)", fontsize=10, loc="left")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].grid(axis="y", color=GRID, lw=.6)
    ax[1].set_axisbelow(True)
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGS / "step2_shrinkage_overview.png", facecolor=SURF)


def validation_figure() -> None:
    r = pd.read_csv(RESULTS / "step2_validation.csv")
    x = r[(r.test == "split_half") & (r.bucket != "all") & (r.method.isin(["per36", "eb_season", "ridge"]))]
    t = x.groupby(["league", "bucket", "method"]).vs_raw_pct.median().unstack("method")
    t.round(2).to_csv(RESULTS / "step2_validation_by_bucket.csv")
    order = ["<100", "100-250", "250-500", "500+"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=200, sharey=True)
    for a, lg in zip(ax, ["NBA", "WNBA"]):
        m = t.xs(lg).reindex(order)
        xs, w = np.arange(len(order)), .27
        for i, (meth, c, lab) in enumerate([("per36", GRAY, "Per-36 (raw per minute)"),
                                            ("eb_season", C1, "Empirical Bayes shrinkage"),
                                            ("ridge", C2, "Ridge (penalty tuned out-of-season)")]):
            a.bar(xs + (i - 1) * w, m[meth], w * .92, color=c, label=lab)
        a.axhline(0, color="#52514e", lw=.8)
        a.set_xticks(xs, order)
        a.set_xlabel("First-half minutes")
        a.set_title(f"{lg}: second-half prediction error vs raw per-possession", fontsize=10, loc="left")
        a.grid(axis="y", color=GRID, lw=.6)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].set_ylabel("Change in error vs raw, % (median over 17 stats)")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGS / "step2_validation.png", facecolor=SURF)


if __name__ == "__main__":
    main()
    validation_figure()
