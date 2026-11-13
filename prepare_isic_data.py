"""Prepare ISIC with INSID3's documented DR-Adapter preprocessing."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
ARCHIVES = {
    "ISIC2018_Task1-2_Training_Input.zip": 11165358566,
    "ISIC2018_Task1_Training_GroundTruth.zip": 27402895,
}
CLASSES = {"seborrheic_keratosis": "1", "nevus": "2", "melanoma": "3"}
EXPECTED_COUNTS = {"1": 208, "2": 1867, "3": 519}
SOURCE_HASHES = {
    "class_id.csv": "2c331fe8d3e789d29d44e684df39ea206cfbb1f5bbb3adfb207c025b69fc6b3e",
    "ISIC_Split.py": "b7e3fc29f73cd053ce8d8e50d364473b15380d048a5371d31758fc110bce9291",
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", type=Path, default=ROOT.parent / "downloads")
    parser.add_argument("--source-dir", type=Path, default=ROOT.parent / "deps/DR-Adapter-isic")
    args = parser.parse_args()
    final = ROOT / "data/ISIC"
    if final.exists():
        parser.error("ISIC already exists; inspect it before running preparation again: " + str(final))
    archives = {}
    for name, size in ARCHIVES.items():
        path = args.download_dir / name
        if not path.exists():
            path = args.download_dir / (name + ".part")
        if not path.is_file() or path.stat().st_size != size:
            parser.error("Missing/incomplete official archive: " + name + "; expected bytes=" + str(size))
        archives[name] = path
    for name, expected in SOURCE_HASHES.items():
        if sha256(args.source_dir / name) != expected:
            raise ValueError("Preprocessing source changed: " + name)
    rows = list(csv.DictReader((args.source_dir / "class_id.csv").open()))
    assert len(rows) == len({r["ID"] for r in rows}) == 2594
    assert Counter(CLASSES[r["Class"]] for r in rows) == EXPECTED_COUNTS
    report = {"source": "INSID3 docs/data.md + DR-Adapter ISIC_Split.py",
              "preprocessing_source_sha256": SOURCE_HASHES, "archives": {},
              "image_preprocessing": "official Image.resize((512,512)) and JPEG save defaults",
              "sampling": "official ISIC loader; 600 episodes across 3 categories, seed=0"}
    # Validate both complete ZIP archives before creating an output dataset.
    for name, path in archives.items():
        print("Checking ZIP CRC:", name, flush=True)
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError("ZIP CRC failed: " + bad)
        report["archives"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    stage_root = ROOT / "data" / ("isic-staging-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    stage = stage_root / "ISIC"
    stage.mkdir(parents=True)
    selected = []
    for name, path in archives.items():
        with zipfile.ZipFile(path) as archive:
            members = [m for m in archive.infolist() if not m.is_dir() and
                       ((name.endswith("Input.zip") and m.filename.endswith(".jpg")) or
                        (name.endswith("GroundTruth.zip") and m.filename.endswith("_segmentation.png")))]
            assert len(members) == 2594, (name, len(members))
            folder = stage / name[:-4]
            folder.mkdir()
            for member in members:
                target = folder / Path(member.filename).name
                with archive.open(member) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
            selected.append(len(members))
    prep = stage_root / "preprocessing"
    (prep / "isic").mkdir(parents=True)
    shutil.copy2(args.source_dir / "class_id.csv", prep / "isic/class_id.csv")
    source = (args.source_dir / "ISIC_Split.py").read_text()
    old = "dir = '../data/ISIC/ISIC/ISIC2018_Task1-2_Training_Input/'"
    assert source.count(old) == 1
    source = source.replace(old, "dir = " + repr(str(stage / "ISIC2018_Task1-2_Training_Input") + "/"))
    patched = prep / "ISIC_Split.py"
    patched.write_text(source)
    subprocess.run([sys.executable, str(patched)], cwd=prep, check=True)
    from PIL import Image
    import numpy as np
    import torch
    from datasets.isic import DatasetISIC
    counts = {}
    for category, expected in EXPECTED_COUNTS.items():
        images = sorted((stage / "ISIC2018_Task1-2_Training_Input" / category).glob("*.jpg"))
        assert len(images) == expected, (category, len(images))
        counts[category] = len(images)
        for image in images:
            with Image.open(image) as im:
                im.load()
                assert im.size == (512, 512)
            mask = stage / "ISIC2018_Task1_Training_GroundTruth" / (image.stem + "_segmentation.png")
            with Image.open(mask) as im:
                im.load()
                assert np.any(np.asarray(im.convert("L")) >= 128)
    ds = DatasetISIC(str(stage_root), shot=1)
    assert len(ds) == 600
    np.random.seed(0)
    torch.manual_seed(0)
    for i in range(3):
        batch = ds[i]
        assert len(batch["ref_imgs"]) == 1
        assert batch["tgt_img"].size == (512, 512)
    # Keep raw originals in the source ZIP, matching the official final layout.
    for path in (stage / "ISIC2018_Task1-2_Training_Input").glob("*.jpg"):
        path.unlink()
    stage.rename(final)
    ds = DatasetISIC(str(ROOT / "data"), shot=1)
    np.random.seed(0)
    cases = []
    for i in range(3):
        target, references, category = ds.sample_episode(i)
        assert target != references[0]
        def mask_of(image):
            return str(final / "ISIC2018_Task1_Training_GroundTruth" / (Path(image).stem + "_segmentation.png"))
        cases.append({"class_id": category, "category": ds.categories[category],
                      "reference_image": references[0], "reference_mask": mask_of(references[0]),
                      "target_image": target, "target_ground_truth_mask": mask_of(target)})
    (ROOT / "examples").mkdir(exist_ok=True)
    (ROOT / "examples/isic-demo-cases.json").write_text(json.dumps({"selection": "official sampler seed=0, first episode per category", "cases": cases}, indent=2) + "\n")
    report.update(status="PASS", images=2594, masks=2594, categories=counts,
                  evaluation_episodes=600, dataset_directory=str(final),
                  patched_preprocessor_sha256=sha256(patched), preprocessing_directory=str(prep))
    destination = ROOT.parent / "setup-logs/isic-data-check.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print("ISIC_DATA_CHECK_PASS", destination, flush=True)


if __name__ == "__main__":
    main()
