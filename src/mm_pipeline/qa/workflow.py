"""Re-export shim: the QA workflow now lives in ``modelvio/workflow.py``.

Kept so ``qa`` and its public API keep importing from here unchanged. Do not add
logic here — edit ``mm_pipeline.modelvio.workflow`` instead
"""

from __future__ import annotations

from mm_pipeline.modelvio.workflow import apply_qa_workflow  # noqa: F401

__all__ = ["apply_qa_workflow"]
