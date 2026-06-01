from pathlib import Path

from mm_pipeline.io import load_dataset_manifest, load_raw_image_manifest


def test_load_dataset_manifest_csv(tmp_path: Path):
    manifest = tmp_path / "datasets.csv"
    manifest.write_text(
        "dataset_id,axis,open_end,approved_labels_dir,images_dir,frame_interval_min\n"
        "trench_a,y,high,/tmp/approved,/tmp/images,1.5\n"
    )

    specs = load_dataset_manifest(manifest)

    assert len(specs) == 1
    spec = specs[0]
    assert spec.dataset_id == "trench_a"
    assert spec.axis == "y"
    assert spec.open_end == "high"
    assert spec.approved_labels_dir == Path("/tmp/approved")
    assert spec.effective_labels_dir == Path("/tmp/approved")
    assert spec.frame_interval_min == 1.5


def test_load_raw_image_manifest_csv(tmp_path: Path):
    manifest = tmp_path / "raw_images.csv"
    manifest.write_text(
        "dataset_id,images_dir,image_pattern,channel,frame_interval_min\n"
        "trench_a,/tmp/images,*.tif,0,1.5\n"
    )

    specs = load_raw_image_manifest(manifest)

    assert len(specs) == 1
    assert specs[0].dataset_id == "trench_a"
    assert specs[0].images_dir == Path("/tmp/images")
    assert specs[0].image_pattern == "*.tif"
    assert specs[0].channel == 0
    assert specs[0].frame_interval_min == 1.5
