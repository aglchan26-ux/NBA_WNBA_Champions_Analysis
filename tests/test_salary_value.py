import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import PROC  # noqa: E402


def test_labels_follow_interval_rule():
    S = pd.read_parquet(PROC / "salary_value.parquet")
    under, over = S.label == "underpaid", S.label == "overpaid"
    assert (S.surplus_lo_s[under] > 0).all()
    assert (S.surplus_hi_s[over] < 0).all()
    assert not (S.label[S.cohort6 == "max_scale"].isin(["underpaid", "overpaid"])).any()
    # surplus is a log difference, never a ratio
    assert np.allclose(S.surplus, S.fair_log - np.log(S.cap_share))


def test_lineups_capture_points():
    v = pd.read_csv(Path(__file__).resolve().parents[1] / "results" / "lineups_validation.csv")
    assert (v.pts_captured.between(0.94, 1.03)).all()
    assert (v.poss_captured.between(0.94, 1.03)).all()
