"""Notebook-first review and correction of selected frame-pair operations.

Currently the pure helpers in this module use two explicit order conventions:

* tracker internals: open-end first;
* compact human notation: closed-end to open-end.

Conversion therefore always reverses the sequence. ``open_end`` determines how
cells are sorted, never whether the operation sequence is reversed. Will change this soon 
to closed-end to open-end ordering for all internals but it's mathematically equivalent
so not urgent.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from mm_pipeline.config import DatasetSpec, TrackerParams
from mm_pipeline.core import (
    CellInstance,
    FramePair,
    TrackingOperation,
    canonical_ops_key,
    deserialize_ops_json,
    serialize_ops_json,
)
from mm_pipeline.io.labels import load_labels_from_folder

from .costs import candidate_ops_cost
from .lineage import reconstruct_lineage
from .select import SelectionResult
from .validation import assert_ops_valid
from .workflow import extract_sorted_cells_for_stack

CorrectionSource = Literal["candidate", "manual"]

_KIND_ALIASES = {
    "l": "link",
    "link": "link",
    "d": "divide",
    "divide": "divide",
    "e": "exit",
    "exit": "exit",
}

CORRECTION_COLUMNS = [
    "dataset_id",
    "pair_id",
    "t",
    "t_next",
    "original_operations",
    "corrected_operations",
    "original_ops_json",
    "corrected_ops_json",
    "source",
    "candidate_rank",
    "dp_cost",
    "note",
]


def parse_compact_kinds(value: str | Sequence[str]) -> tuple[str, ...]:
    """Parse closed-to-open notation into full operation-kind names."""

    tokens: list[str]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Operation sequence must not be empty.")
        bare = re.sub(r"[\s,()\[\]{}'\"]", "", text).lower()
        if bare and set(bare) <= {"l", "d", "e"}:
            tokens = list(bare)
        else:
            stripped = re.sub(r"[()\[\]{}'\"]", " ", text.lower())
            tokens = [token for token in re.split(r"[\s,]+", stripped) if token]
    else:
        tokens = [str(token).strip().lower() for token in value]
    if not tokens:
        raise ValueError("Operation sequence must not be empty.")

    kinds: list[str] = []
    for token in tokens:
        if token not in _KIND_ALIASES:
            raise ValueError(
                f"Unknown operation token {token!r}; use l/link, d/divide, or e/exit."
            )
        kinds.append(_KIND_ALIASES[token])
    return tuple(kinds)


def format_compact_ops(ops: Sequence[TrackingOperation | Sequence[object]]) -> str:
    """Format internal open-first labelled ops as closed-to-open letters."""

    normalised = [
        op if isinstance(op, TrackingOperation) else TrackingOperation.from_tuple(op)
        for op in ops
    ]
    return "".join(op.kind[0] for op in reversed(normalised))


def infer_ops_from_compact(
    value: str | Sequence[str],
    cells_t: Sequence[CellInstance],
    cells_k: Sequence[CellInstance],
) -> tuple[TrackingOperation, ...]:
    """Expand compact kinds to labelled internal operations.

    ``cells_t`` and ``cells_k`` must already be in internal open-end-first order.
    User-facing exits form a suffix; after reversal this is the tracker's exit prefix.
    """

    human_kinds = parse_compact_kinds(value)
    if len(human_kinds) != len(cells_t):
        raise ValueError(
            f"Expected {len(cells_t)} operations for frame t, got {len(human_kinds)}."
        )
    seen_exit = False
    for kind in human_kinds:
        if kind == "exit":
            seen_exit = True
        elif seen_exit:
            raise ValueError(
                "In closed-to-open notation, exit operations must form the open-end suffix."
            )

    internal_kinds = tuple(reversed(human_kinds))
    dest_index = 0
    ops: list[TrackingOperation] = []
    for source, kind in zip(cells_t, internal_kinds):
        if kind == "exit":
            ops.append(TrackingOperation("exit", int(source.label)))
        elif kind == "link":
            if dest_index >= len(cells_k):
                raise ValueError("Link sequence consumes more destination cells than exist.")
            ops.append(
                TrackingOperation("link", int(source.label), int(cells_k[dest_index].label))
            )
            dest_index += 1
        elif kind == "divide":
            if dest_index + 1 >= len(cells_k):
                raise ValueError("Divide sequence consumes more destination cells than exist.")
            ops.append(
                TrackingOperation(
                    "divide",
                    int(source.label),
                    int(cells_k[dest_index].label),
                    int(cells_k[dest_index + 1].label),
                )
            )
            dest_index += 2

    if dest_index != len(cells_k):
        raise ValueError(
            f"Operation sequence consumes {dest_index} destination cells; "
            f"frame t_next contains {len(cells_k)}."
        )
    assert_ops_valid(cells_t, cells_k, ops)
    return tuple(ops)


def find_candidate_by_ops(pair_candidates: Any, ops: Sequence[TrackingOperation]) -> Any | None:
    """Return the exact candidate row matching ``ops``, or ``None``."""

    target = canonical_ops_key(ops)
    for idx, row in pair_candidates.iterrows():
        row_ops = deserialize_ops_json(str(row["ops_json"]))
        if canonical_ops_key(row_ops) == target:
            matched = row.copy()
            matched.name = idx
            return matched
    return None


def apply_selection_overrides(
    selections: Sequence[SelectionResult],
    overrides: Mapping[int, "TrackingCorrection"],
) -> list[SelectionResult]:
    """Return new immutable selections with corrected ops substituted by frame."""

    out: list[SelectionResult] = []
    seen: set[int] = set()
    for selection in selections:
        correction = overrides.get(int(selection.t))
        if correction is None:
            out.append(selection)
            continue
        seen.add(int(selection.t))
        out.append(
            replace(
                selection,
                chosen_ops_json=correction.corrected_ops_json,
                chosen_idx=correction.candidate_index,
                chosen_score=float(correction.dp_cost),
            )
        )
    missing = set(int(t) for t in overrides) - seen
    if missing:
        raise KeyError(f"No original selections for corrected frame(s): {sorted(missing)}.")
    return out


@dataclass(frozen=True)
class TrackingCorrection:
    dataset_id: str
    pair_id: str
    t: int
    t_next: int
    original_ops_json: str
    corrected_ops_json: str
    source: CorrectionSource
    candidate_rank: int | None
    dp_cost: float
    note: str = ""
    candidate_index: Any | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "pair_id": self.pair_id,
            "t": self.t,
            "t_next": self.t_next,
            "original_operations": format_compact_ops(
                deserialize_ops_json(self.original_ops_json)
            ),
            "corrected_operations": format_compact_ops(
                deserialize_ops_json(self.corrected_ops_json)
            ),
            "original_ops_json": self.original_ops_json,
            "corrected_ops_json": self.corrected_ops_json,
            "source": self.source,
            "candidate_rank": self.candidate_rank,
            "dp_cost": self.dp_cost,
            "note": self.note,
        }


def corrections_to_dataframe(corrections: Sequence[TrackingCorrection]) -> Any:
    import pandas as pd

    if not corrections:
        return pd.DataFrame(columns=CORRECTION_COLUMNS)
    return pd.DataFrame([item.to_row() for item in corrections], columns=CORRECTION_COLUMNS)


class TrackingReviewSession:
    """Thin stateful notebook wrapper around the pure correction helpers."""

    def __init__(
        self,
        spec: DatasetSpec,
        candidates: Any,
        selection_result: Any,
        *,
        tracker_params: TrackerParams | None = None,
    ) -> None:
        import pandas as pd

        self.spec = spec
        if tracker_params is None:
            resolved_config = getattr(candidates, "resolved_config", {})
            stored_params = (
                resolved_config.get("tracker")
                if isinstance(resolved_config, Mapping)
                else None
            )
            if not isinstance(stored_params, Mapping):
                raise ValueError(
                    "tracker_params is required for a candidate table or path; "
                    "pass the exact parameters used to generate the candidates."
                )
            tracker_params = TrackerParams.from_mapping(stored_params)
        self.tracker_params = tracker_params
        if self.tracker_params.axis != spec.axis:
            raise ValueError("tracker_params.axis must match spec.axis.")
        labels_dir = spec.effective_labels_dir
        if labels_dir is None:
            raise ValueError("Tracking review requires spec.effective_labels_dir.")
        self.labels = load_labels_from_folder(labels_dir)
        self.cells_by_frame = extract_sorted_cells_for_stack(
            self.labels,
            dataset_id=spec.dataset_id,
            axis=spec.axis,
            open_end=spec.open_end,
        )

        table = candidates.candidates_df if hasattr(candidates, "candidates_df") else candidates
        if isinstance(table, (str, Path)):
            table = pd.read_parquet(table)
        if not isinstance(table, pd.DataFrame):
            raise TypeError("candidates must be a DataFrame, parquet path, or TrackGenerateResult.")
        required = {"dataset_id", "t", "pair_id", "ops_json"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"Candidate table is missing columns: {sorted(missing)}.")
        self.candidates_df = table.loc[table["dataset_id"] == spec.dataset_id].copy()
        if self.candidates_df.empty:
            raise ValueError(f"No candidates for dataset {spec.dataset_id!r}.")

        if hasattr(selection_result, "selections_by_dataset"):
            try:
                selections = selection_result.selections_by_dataset[spec.dataset_id]
            except KeyError as exc:
                raise KeyError(
                    f"Selection result has no dataset {spec.dataset_id!r}."
                ) from exc
        else:
            selections = selection_result
        self.original_selections = tuple(selections)
        self._selection_by_t = {int(selection.t): selection for selection in self.original_selections}
        self._overrides: dict[int, TrackingCorrection] = {}
        self._validate_candidate_geometry()

    def _pair_rows(self, t: int) -> Any:
        pair = self.candidates_df.loc[self.candidates_df["t"].astype(int) == int(t)].copy()
        if pair.empty:
            raise KeyError(f"No candidates for frame pair ({int(t)},{int(t) + 1}).")
        sort_col = "sample_rank" if "sample_rank" in pair.columns else "dp_rank_global"
        if sort_col in pair.columns:
            pair = pair.sort_values(sort_col)
        return pair

    def _cells_for_t(self, t: int) -> tuple[tuple[CellInstance, ...], tuple[CellInstance, ...]]:
        ti = int(t)
        if ti < 0 or ti + 1 >= len(self.cells_by_frame):
            raise IndexError(f"Frame pair ({ti},{ti + 1}) is outside the label stack.")
        return self.cells_by_frame[ti], self.cells_by_frame[ti + 1]

    def _frame_pair(self, t: int) -> FramePair:
        return FramePair(
            dataset_id=self.spec.dataset_id,
            t=int(t),
            k=int(t) + 1,
            frame_shape=(int(self.labels.shape[1]), int(self.labels.shape[2])),
            axis=self.spec.axis,  # type: ignore[arg-type]
            open_end=self.spec.open_end,  # type: ignore[arg-type]
        )

    def _validate_candidate_geometry(self) -> None:
        labels_dir = str(Path(self.spec.effective_labels_dir).resolve())
        if "labels_dir" in self.candidates_df.columns:
            recorded = {
                str(Path(str(value)).resolve())
                for value in self.candidates_df["labels_dir"].dropna().unique()
                if str(value)
            }
            if recorded and recorded != {labels_dir}:
                raise ValueError(
                    f"Candidate labels_dir {sorted(recorded)} does not match spec labels_dir {labels_dir!r}."
                )
        for _, row in self.candidates_df.iterrows():
            t = int(row["t"])
            delta_t = int(row.get("delta_t", 1))
            if delta_t != 1:
                raise ValueError("Tracking review currently supports adjacent frame pairs only.")
            cells_t, cells_k = self._cells_for_t(t)
            try:
                assert_ops_valid(cells_t, cells_k, deserialize_ops_json(str(row["ops_json"])))
            except ValueError as exc:
                raise ValueError(
                    f"Candidate geometry no longer matches labels for pair ({t},{t + 1}): {exc}"
                ) from exc

    @staticmethod
    def _rank_from_row(row: Any) -> int | None:
        for name in ("dp_rank_global", "sample_rank"):
            if name in row.index:
                value = row[name]
                try:
                    if value is not None and not math.isnan(float(value)):
                        return int(value)
                except (TypeError, ValueError):
                    pass
        return None

    def find_candidate(self, t: int, operations: str | Sequence[str]) -> Any | None:
        cells_t, cells_k = self._cells_for_t(t)
        ops = infer_ops_from_compact(operations, cells_t, cells_k)
        return find_candidate_by_ops(self._pair_rows(t), ops)

    def pair_candidates(self, t: int, limit: int | None = None) -> Any:
        import pandas as pd

        pair = self._pair_rows(t)
        if limit is not None:
            if int(limit) < 0:
                raise ValueError("limit must be non-negative or None.")
            pair = pair.head(int(limit))
        current_json = (
            self._overrides[int(t)].corrected_ops_json
            if int(t) in self._overrides
            else self._selection_by_t[int(t)].chosen_ops_json
        )
        current_key = (
            canonical_ops_key(deserialize_ops_json(current_json)) if current_json is not None else None
        )
        min_cost = (
            float(pair["dp_cost"].astype(float).min())
            if "dp_cost" in pair.columns and pair["dp_cost"].notna().any()
            else float("nan")
        )
        rows: list[dict[str, Any]] = []
        for _, row in pair.iterrows():
            ops = deserialize_ops_json(str(row["ops_json"]))
            cost = float(row["dp_cost"]) if "dp_cost" in row and pd.notna(row["dp_cost"]) else float("nan")
            item = {
                "rank": self._rank_from_row(row),
                "dp_cost": cost,
                "cost_delta": cost - min_cost if math.isfinite(cost) and math.isfinite(min_cost) else float("nan"),
                "operations": format_compact_ops(ops),
                "n_links": sum(op.kind == "link" for op in ops),
                "n_divides": sum(op.kind == "divide" for op in ops),
                "n_exits": sum(op.kind == "exit" for op in ops),
                "selected": canonical_ops_key(ops) == current_key,
            }
            for col in ("raw_score", "pair_score_rank", "score_rank"):
                if col in pair.columns:
                    item[col] = row[col]
            rows.append(item)
        return pd.DataFrame(rows)

    def _set_override(
        self,
        t: int,
        ops: Sequence[TrackingOperation],
        *,
        source: CorrectionSource,
        candidate_row: Any | None,
        note: str | None,
    ) -> TrackingCorrection:
        ti = int(t)
        original = self._selection_by_t.get(ti)
        if original is None or original.chosen_ops_json is None:
            raise KeyError(f"No original kept selection for frame pair ({ti},{ti + 1}).")
        pair = self._frame_pair(ti)
        cells_t, cells_k = self._cells_for_t(ti)
        corrected_json = serialize_ops_json(ops)
        if candidate_row is not None:
            rank = self._rank_from_row(candidate_row)
            cost = float(candidate_row["dp_cost"])
            candidate_index = candidate_row.name
        else:
            rank = None
            cost = candidate_ops_cost(
                cells_t, cells_k, pair, self.tracker_params, ops
            )
            candidate_index = None
        correction = TrackingCorrection(
            dataset_id=self.spec.dataset_id,
            pair_id=pair.pair_id,
            t=ti,
            t_next=ti + 1,
            original_ops_json=str(original.chosen_ops_json),
            corrected_ops_json=corrected_json,
            source=source,
            candidate_rank=rank,
            dp_cost=float(cost),
            note="" if note is None else str(note),
            candidate_index=candidate_index,
        )
        self._overrides[ti] = correction
        return correction

    def select_candidate(
        self, t: int, rank: int, note: str | None = None
    ) -> TrackingCorrection:
        pair = self._pair_rows(t)
        matches = [
            row
            for _, row in pair.iterrows()
            if self._rank_from_row(row) == int(rank)
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Expected one candidate at rank {int(rank)} for pair ({int(t)},{int(t) + 1}); "
                f"found {len(matches)}."
            )
        row = matches[0]
        return self._set_override(
            int(t),
            deserialize_ops_json(str(row["ops_json"])),
            source="candidate",
            candidate_row=row,
            note=note,
        )

    def select_manual(
        self,
        t: int,
        operations: str | Sequence[str],
        note: str | None = None,
    ) -> TrackingCorrection:
        cells_t, cells_k = self._cells_for_t(t)
        ops = infer_ops_from_compact(operations, cells_t, cells_k)
        matched = find_candidate_by_ops(self._pair_rows(t), ops)
        return self._set_override(
            int(t),
            ops,
            source="manual",
            candidate_row=matched,
            note=note,
        )

    def clear_correction(self, t: int) -> None:
        self._overrides.pop(int(t), None)

    @property
    def corrections(self) -> Any:
        return corrections_to_dataframe(
            [self._overrides[t] for t in sorted(self._overrides)]
        )

    def reconstruct(
        self,
        *,
        out_dir: str | Path | None = None,
        run_tag: str = "tracking_corrected",
        overwrite: bool = False,
    ) -> Any:
        """Reconstruct from frame zero and optionally save a new corrected run."""

        from mm_pipeline.io.tracks import write_lineage_outputs
        from mm_pipeline.runners._outputs import make_run_metadata, write_multifile_outputs
        from mm_pipeline.runners.track_select import TrackSelectResult

        selections = apply_selection_overrides(self.original_selections, self._overrides)
        tracks, events, divisions = reconstruct_lineage(
            selections,
            self.labels,
            open_end=self.spec.open_end,
            axis=self.spec.axis,
        )

        output_dir: Path | None = None
        if out_dir is not None:
            output_dir = Path(out_dir) / str(run_tag)
            if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
                raise FileExistsError(
                    f"{output_dir} is not empty. Pass overwrite=True or choose a new run_tag."
                )
            dataset_dir = output_dir / self.spec.dataset_id
            write_lineage_outputs(tracks, events, divisions, dataset_dir)
            self.corrections.to_csv(dataset_dir / "corrections.csv", index=False)
            metadata = make_run_metadata(
                command="tracking-review",
                manifest_path=None,
                resolved_config={
                    "tracker": self.tracker_params.to_dict(),
                    "run_tag": str(run_tag),
                },
                dataset_ids=[self.spec.dataset_id],
            )
            write_multifile_outputs(
                out_dir=output_dir,
                summary={
                    "n_datasets": 1,
                    "n_corrections_total": len(self._overrides),
                    "n_selections_total": len(selections),
                },
                metadata=metadata,
                title="tracking review run",
                overwrite=overwrite,
            )

        return TrackSelectResult(
            tracks_by_dataset={self.spec.dataset_id: tracks},
            events_by_dataset={self.spec.dataset_id: events},
            divisions_by_dataset={self.spec.dataset_id: divisions},
            selections_by_dataset={self.spec.dataset_id: selections},
            resolved_config={
                "tracker": self.tracker_params.to_dict(),
                "n_corrections": len(self._overrides),
            },
            output_dir=output_dir,
        )


__all__ = [
    "CORRECTION_COLUMNS",
    "TrackingCorrection",
    "TrackingReviewSession",
    "apply_selection_overrides",
    "corrections_to_dataframe",
    "find_candidate_by_ops",
    "format_compact_ops",
    "infer_ops_from_compact",
    "parse_compact_kinds",
]
