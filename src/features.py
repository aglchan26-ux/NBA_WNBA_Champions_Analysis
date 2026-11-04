"""Feature definitions: every feature is (numerator, exposure, model)."""
import pandas as pd

from common import PROC

ZONES = ["rim", "short_mid", "long_mid", "corner3", "above3"]

# name: (numerator expression, exposure column, model, plain-language label)
RATE = {
    "usage":    ("fga + 0.44 * fta + tov", "poss_on", "Shots + FT trips + turnovers per possession on floor"),
    "fta_rate": ("fta", "poss_on", "Free-throw attempts per possession"),
    "pts":      ("pts", "poss_on", "Points per possession"),
    "ast":      ("ast", "tm_fgm_on", "Assists per teammate made shot on floor"),
    "orb":      ("orb", "orb_chances_on", "Offensive rebounds per available offensive rebound"),
    "drb":      ("drb", "drb_chances_on", "Defensive rebounds per available defensive rebound"),
    "stl":      ("stl", "poss_on", "Steals per possession"),
    "blk":      ("blk", "opp_fga_on", "Blocks per opponent shot attempt"),
    "tov":      ("tov", "poss_on", "Turnovers per possession"),
    "pf":       ("pf", "poss_on", "Personal fouls per possession"),
}
PROP = {
    "fg3_rate": ("fg3a", "fga", "Share of shots that are threes"),
    "assisted": ("assisted_fgm", "pbp_fgm", "Share of made shots that were assisted"),
    "fg3_pct":  ("fg3m", "fg3a", "Three-point %"),
    "ft_pct":   ("ftm", "fta", "Free-throw %"),
    "rim_pct":  ("rim_fgm", "rim_fga", "FG% at the rim"),
    "mid_pct":  ("mid_fgm", "mid_fga", "FG% from midrange (4 ft to the arc)"),
}


def load(season_type: str = "regular", source: pd.DataFrame | None = None) -> pd.DataFrame:
    """Player-season counts + exposures for one season type (or a custom aggregated frame)."""
    if source is None:
        ps = pd.read_parquet(PROC / "player_season.parquet")
        pz = pd.read_parquet(PROC / "player_zones.parquet")
        d = ps[ps.season_type == season_type].merge(
            pz, on=["league", "season", "season_type", "athlete_id"], how="left")
    else:
        d = source.copy()
    for z in ZONES:
        for s in ("fga", "fgm"):
            col = f"{z}_{s}"
            d[col] = d[col].fillna(0) if col in d else 0
    for c in ["assisted_fgm", "pbp_fgm", "pbp_fga"]:
        d[c] = d[c].fillna(0) if c in d else 0
    d["mid_fga"] = d.short_mid_fga + d.long_mid_fga
    d["mid_fgm"] = d.short_mid_fgm + d.long_mid_fgm
    for name, (expr, _, _) in RATE.items():
        d[f"{name}_num"] = d.eval(expr)
    return d
