# Example data

A small synthetic mother-machine trench generated with [SyMBac](https://github.com/georgeoshardo/SyMBac), used by [`../notebooks/example.ipynb`](../notebooks/example.ipynb).

- `labels/` — 791 per-frame label TIFFs (`Nonesynth_NNNNN.tiff`)
- `gt_tracks.csv` — ground-truth tracks (`track_id, t, label, x, y, area, axis_len`)
- `gt_divisions.csv` — ground-truth division events (`t_div, mother_track_id, d1_track_id, d2_track_id`)
