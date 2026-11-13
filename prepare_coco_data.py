"""Prepare every asset referenced by the official four COCO-20i validation folds."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DOWNLOADS = ROOT.parent / "downloads"
LOGS = ROOT.parent / "setup-logs"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    image_report = json.loads((LOGS / "coco-images-download-status.json").read_text())
    assert image_report["status"] == "completed" and image_report["zip_crc"] == "PASS"
    assets = json.loads((DOWNLOADS / "coco-local-assets-check.json").read_text())
    records = {}
    for asset in assets:
        path = DOWNLOADS / asset["file"]
        assert path.stat().st_size == asset["bytes"]
        assert asset["zip_crc"] == "PASS" and digest(path) == asset["sha256"]
        records[asset["file"]] = asset
    image_zip = DOWNLOADS / "coco-val2014-images.zip"
    assert image_zip.stat().st_size == image_report["bytes"]
    records[image_zip.name] = {k: image_report[k] for k in ["bytes", "sha256", "zip_crc", "url"]}
    final = ROOT / "data/COCO2014"
    assert not final.exists(), "Inspect the existing COCO2014 directory before rerunning."
    stage_parent = ROOT / "data" / ("coco-staging-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    stage = stage_parent / "COCO2014"
    for folder in ["val2014", "annotations/val2014", "splits/val", "splits/trn", "example_masks"]:
        (stage / folder).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DOWNLOADS / "coco-splits.zip") as z:
        for member in z.infolist():
            parts = Path(member.filename).parts
            if member.is_dir() or len(parts) < 3 or parts[-3] != "splits" or parts[-2] not in ("val", "trn") or not parts[-1].endswith(".pkl"):
                continue
            (stage / "splits" / parts[-2] / parts[-1]).write_bytes(z.read(member))
    metadata = []
    required = set()
    for fold in range(4):
        with (stage / "splits/val" / f"fold{fold}.pkl").open("rb") as stream:
            classes = pickle.load(stream)  # Author-provided pickle from the official linked source.
        nonempty = {int(k) for k, values in classes.items() if values}
        assert nonempty == set(range(fold, 80, 4)), (fold, nonempty)
        assert all(len(set(classes[c])) >= 2 for c in nonempty)
        names = {name for values in classes.values() for name in values}
        assert all(len(Path(name).parts) == 2 and Path(name).parts[0] == "val2014" for name in names)
        required.update(names)
        metadata.append(classes)
    counts = {}
    for archive, destination, suffix in [(image_zip, stage / "val2014", ".jpg"),
                                         (DOWNLOADS / "coco-val2014-masks.zip", stage / "annotations/val2014", ".png")]:
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.infolist() if not m.is_dir() and m.filename.endswith(suffix)
                       and Path(m.filename).parts[-2] == "val2014"]
            assert members
            for member in members:
                path = destination / Path(member.filename).name
                assert not path.exists(), member.filename
                with z.open(member) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target)  # zipfile also verifies each extracted entry's CRC.
            counts[suffix] = len(members)
    for index, name in enumerate(sorted(required)):
        image_path = stage / name
        mask_path = stage / "annotations" / (name[:-4] + ".png")
        assert image_path.is_file() and mask_path.is_file(), name
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            assert image.size == mask.size and len(mask.getbands()) == 1, name
            if index % 197 == 0:
                values = np.asarray(mask)
                assert values.ndim == 2 and set(np.unique(values).tolist()) <= set(range(81)) | {255}
    from datasets.coco import DatasetCOCO
    folds = []
    cases = []
    for fold in range(4):
        ds = DatasetCOCO(str(stage_parent), fold=fold, shot=1)
        assert len(ds) == 1000 and len(ds.class_ids) == 20
        # Exactly mirror the first official sample, without consuming the evaluation process's RNG.
        np.random.seed(0)
        category = int(np.random.choice(ds.class_ids, 1, replace=False)[0])
        target = np.random.choice(ds.img_metadata_classwise[category], 1, replace=False)[0]
        while True:
            reference = np.random.choice(ds.img_metadata_classwise[category], 1, replace=False)[0]
            if reference != target:
                break
        for name, role in [(target, "target"), (reference, "reference")]:
            raw = np.asarray(Image.open(stage / "annotations" / (name[:-4] + ".png")))
            Image.fromarray((raw == category+1).astype(np.uint8)*255).save(stage / "example_masks" / f"fold{fold}_{role}.png")
        np.random.seed(0)
        batch = ds[0]
        assert int(batch["class_id"]) == category and batch["tgt_mask"].ndim == 2
        assert len(batch["ref_imgs"]) == 1 and batch["ref_masks"][0].ndim == 2
        cases.append({"fold": fold, "class_id": category,
                      "reference_image": str(final / reference),
                      "reference_mask": str(final / "example_masks" / f"fold{fold}_reference.png"),
                      "target_image": str(final / target),
                      "target_ground_truth_mask": str(final / "example_masks" / f"fold{fold}_target.png")})
        folds.append({"fold": fold, "episodes": len(ds), "class_ids": list(ds.class_ids),
                      "unique_images": len(ds.img_metadata),
                      "reference_pool_size_per_class": {int(c): len(ds.img_metadata_classwise[c]) for c in ds.class_ids}})
    stage.rename(final)
    (ROOT / "examples/coco-demo-cases.json").write_text(json.dumps({"selection": "seed=0, first official sample per fold", "cases": cases}, indent=2) + "\n")
    report = {"status": "PASS", "dataset": "COCO-20i", "images": counts[".jpg"], "masks": counts[".png"],
              "unique_images_referenced_by_validation_folds": len(required), "folds": folds,
              "archives": records, "dataset_directory": str(final),
              "scope": "all assets referenced by all four official validation folds; no training images required",
              "preprocessing": "original images and indexed masks unchanged; only model's official resize",
              "checks": "source ZIP CRC; all referenced paths and image/mask dimensions; sampled pixel labels; official loader samples",
              "source": "https://github.com/visinf/INSID3/blob/main/docs/data.md"}
    (LOGS / "coco-data-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print("COCO_DATA_CHECK_PASS", json.dumps(folds), flush=True)


if __name__ == "__main__":
    main()
