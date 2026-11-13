"""Prepare the exact iSAID-5i patches and folds linked by official INSID3."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    downloads = ROOT.parent / "downloads"
    sources = [downloads / "iSAID_5i.zip", downloads / "isaid-splits.zip"]
    final = ROOT / "data/iSAID"
    assert not final.exists(), "Inspect the existing iSAID directory before rerunning."
    archive_records = {}
    for source in sources:
        with zipfile.ZipFile(source) as z:
            bad = z.testzip()
            assert bad is None, (source.name, bad)
        archive_records[source.name] = {"bytes": source.stat().st_size, "sha256": digest(source)}
    stage = ROOT / "data" / ("iSAID.preparing-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    for name in ["val/images", "val/semantic_png", "splits/val"]:
        (stage / name).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(sources[0]) as z:
        counts = Counter()
        for member in z.infolist():
            parts = Path(member.filename).parts
            if member.is_dir() or not member.filename.endswith(".png") or len(parts) < 3:
                continue
            if parts[-3:-1] not in [("val", "images"), ("val", "semantic_png")]:
                continue
            folder = parts[-2]
            destination = stage / "val" / folder / parts[-1]
            assert not destination.exists(), "Duplicate basename: " + member.filename
            with z.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            counts[folder] += 1
    assert counts == {"images": 6363, "semantic_png": 6363}, counts
    with zipfile.ZipFile(sources[1]) as z:
        for fold in range(3):
            matches = [n for n in z.namelist() if Path(n).parts[-3:] == ("splits", "val", f"fold{fold}.txt")]
            assert len(matches) == 1, matches
            payload = z.read(matches[0])
            assert payload.endswith(b"\n"), "Official loader expects a trailing newline."
            (stage / "splits/val" / f"fold{fold}.txt").write_bytes(payload)
    mask_labels = {}
    all_labels = set()
    ignore_pixels = 0
    for image in sorted((stage / "val/images").glob("*.png")):
        mask = stage / "val/semantic_png" / (image.stem + "_instance_color_RGB.png")
        assert mask.is_file(), mask
        with Image.open(image) as im:
            im.load()
            size = im.size
        with Image.open(mask) as im:
            assert im.size == size
            # Preserve palette/class indices; converting these masks to RGB is wrong.
            array = np.asarray(im)
        assert array.ndim == 2, (mask, array.shape)
        labels = set(np.unique(array).tolist())
        assert labels <= set(range(16)) | {255}, (mask, labels)
        mask_labels[image.stem] = labels
        all_labels.update(labels)
        ignore_pixels += int((array == 255).sum())
    from datasets.isaid import DatasetISAID
    folds = []
    cases = []
    absent_target_annotations = []
    masks = stage / "example_masks"
    masks.mkdir()
    for fold in range(3):
        split = (stage / "splits/val" / f"fold{fold}.txt").read_text()
        rows = [line for line in split.splitlines() if line]
        per_class = Counter()
        for row in rows:
            name, raw_class = row.split("__")
            identity = int(raw_class) - 1
            assert fold * 5 <= identity < (fold + 1) * 5
            assert name in mask_labels, (fold, row)
            if int(raw_class) not in mask_labels[name]:
                absent_target_annotations.append({"fold": fold, "episode": row})
            per_class[identity] += 1
        assert set(per_class) == set(range(fold * 5, (fold + 1) * 5))
        ds = DatasetISAID(str(stage), fold=fold, shot=1)
        assert len(ds) == len(rows)
        assert all(len(set(ds.img_metadata_classwise[c])) >= 2 for c in ds.class_ids)
        np.random.seed(0)
        target, references, identity = ds.sample_episode(0)
        assert target != references[0]
        # Export binary masks only for qualitative examples; evaluation reads original indexed masks.
        for name, role in [(target, "target"), (references[0], "reference")]:
            raw = np.asarray(Image.open(stage / "val/semantic_png" / (name + "_instance_color_RGB.png")))
            Image.fromarray((raw == identity + 1).astype(np.uint8) * 255).save(masks / f"fold{fold}_{role}.png")
        cases.append({"fold": fold, "class_id": identity,
                      "reference_image": str(final / "val/images" / (references[0] + ".png")),
                      "reference_mask": str(final / "example_masks" / f"fold{fold}_reference.png"),
                      "target_image": str(final / "val/images" / (target + ".png")),
                      "target_ground_truth_mask": str(final / "example_masks" / f"fold{fold}_target.png")})
        np.random.seed(0)
        batch = ds[0]
        assert batch["tgt_mask"].ndim == batch["tgt_ignore_idx"].ndim == 2
        assert batch["tgt_mask"].shape == batch["tgt_ignore_idx"].shape
        assert not np.any((batch["tgt_mask"].numpy() > 0) & (batch["tgt_ignore_idx"].numpy() > 0))
        folds.append({"fold": fold, "episodes": len(ds), "class_ids": list(ds.class_ids),
                      "episodes_per_class": dict(per_class)})
    stage.rename(final)
    (ROOT / "examples/isaid-demo-cases.json").write_text(json.dumps({"selection": "seed=0, first official episode in each fold", "cases": cases}, indent=2) + "\n")
    report = {"status": "PASS", "dataset": "iSAID-5i", "images": 6363, "masks": 6363,
              "mask_labels": sorted(all_labels), "ignore_pixels": ignore_pixels,
              "metadata_targets_without_foreground": absent_target_annotations,
              "archives": archive_records, "folds": folds, "dataset_directory": str(final),
              "source": "https://github.com/visinf/INSID3/blob/main/docs/data.md",
              "preprocessing": "official pre-cut patches; no resizing, recoloring or new splits"}
    (ROOT.parent / "setup-logs/isaid-data-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print("ISAID_DATA_CHECK_PASS", json.dumps(folds), flush=True)


if __name__ == "__main__":
    main()
