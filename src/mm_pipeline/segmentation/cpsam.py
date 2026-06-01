"""CPSAM segmentation backend

The Cellpose/CPSAM imports are lazy so users can run the
tracking side of the package with precomputed labels and no Cellpose install.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from mm_pipeline.config import SegmentationConfig, SegmentationRunArtifact
from mm_pipeline.io.images import read_image

from .base import utc_now_iso, write_segmentation_metadata
from .overlays import save_mask_overlays
from .validation import validate_label_directory


class CPSAMBackend:
    name = "cpsam"

    def segment_images(
        self,
        image_paths: Sequence[str | Path],
        output_dir: str | Path,
        config: SegmentationConfig,
        *,
        dataset_id: str,
    ) -> SegmentationRunArtifact:
        """Run CPSAM/Cellpose over image paths.

        Tests should not call this method unless explicitly exercising a real
        Cellpose installation and accepting the runtime cost.
        """

        try:
            from cellpose import io, models
            import numpy as np
            import tifffile as tiff
        except ImportError as exc:
            raise RuntimeError("CPSAM segmentation requires cellpose and tifffile.") from exc

        paths = [Path(p) for p in image_paths]
        if not paths:
            raise ValueError("CPSAMBackend requires at least one input image.")

        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        label_dir = out_root / "label_tifs"
        label_dir.mkdir(parents=True, exist_ok=True)
        filled_dir = out_root / "overlays_filled"
        outlines_dir = out_root / "overlays_outlines"
        if config.overlays:
            filled_dir.mkdir(parents=True, exist_ok=True)
            outlines_dir.mkdir(parents=True, exist_ok=True)

        model = models.CellposeModel(gpu=config.use_gpu, model_type=config.model_type)

        for path in paths:
            base = path.stem
            img = io.imread(path)
            masks, flows, _styles = model.eval(
                [img],
                channels=[config.chan, config.chan2],
                flow_threshold=config.flow_threshold,
                cellprob_threshold=config.cellprob_threshold,
            )
            mask = masks[0] if isinstance(masks, list) else masks
            label_path = label_dir / f"{base}.tif"
            tiff.imwrite(label_path, np.asarray(mask, dtype=np.uint32))

            if config.save_pngs:
                out_file = out_root / path.name
                io.save_masks(
                    images=[img],
                    masks=masks,
                    flows=flows,
                    file_names=[str(out_file)],
                    channels=[config.chan, config.chan2],
                    png=True,
                    tif=False,
                    save_txt=False,
                    save_flows=False,
                    save_outlines=False,
                )

            if config.overlays:
                save_mask_overlays(read_image(path), np.asarray(mask), base_name=base, filled_dir=filled_dir, outlines_dir=outlines_dir)

        validation = validate_label_directory(label_dir)
        validation.raise_for_errors()
        artifact = SegmentationRunArtifact(
            dataset_id=dataset_id,
            backend=self.name,
            model_type=config.model_type,
            raw_images_dir=paths[0].parent,
            label_tifs_dir=label_dir,
            overlays_filled_dir=filled_dir if config.overlays else None,
            overlays_outlines_dir=outlines_dir if config.overlays else None,
            config=config.to_dict(),
            image_count=len(paths),
            label_count=validation.frame_count,
            frame_shape=validation.frame_shape,
            created_at=utc_now_iso(),
            metadata={"validation": validation.__dict__},
        )
        write_segmentation_metadata(artifact, out_root)
        return artifact
