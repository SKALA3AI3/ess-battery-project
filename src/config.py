"""Shared configuration for ESS battery experiments."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT.parent

DEFAULT_DATA_CANDIDATES = [
    PROJECT_ROOT / "feat_all.csv",
    MODEL_ROOT / "data" / "feat_all.csv",
]

BAD_CELLS = [
    "Batch 1_cell08",
    "Batch 1_cell10",
    "Batch 1_cell12",
    "Batch 1_cell13",
    "Batch 1_cell22",
]

RANDOM_SEED = 42
CV_FOLDS = 5
DEFAULT_TEST_SIZE = 0.20
TARGET_LOG = "log_cycle_life"
TARGET_RAW = "cycle_life"
TARGET_MAPE = 9.1

