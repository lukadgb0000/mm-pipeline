"""Re-export shim: per-pair aggregation now lives in ``modelvio/aggregation.py``.

Kept so ``qa`` and its public API keep importing from here unchanged. Do not add
logic here — edit ``mm_pipeline.modelvio.aggregation`` instead
"""

from __future__ import annotations

from mm_pipeline.modelvio.aggregation import (  # noqa: F401
    PAIR_ID_COLS,
    build_per_pair_features,
    per_pair_feature_columns,
)

__all__ = ["PAIR_ID_COLS", "build_per_pair_features", "per_pair_feature_columns"]
