"""Prepare CWFID vegetation masks and a disjoint reference/test protocol."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
EXPECTED_SHA = "ddb5b12ec5836bcaa70e6402cd4b6083ff858695165e407c53a0a60b153e53ce"


def main():
    archive = ROOT.parent / "downloads/cwfid-master.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == EXPECTED_SHA
    final = ROOT / "data/CWFID"
    assert not final.exists(), "Inspect the existing CWFID directory before rerunning."
    stage = ROOT / "data/CWFID.preparing"
    stage.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        prefix = z.namelist()[0].split("/")[0] + "/"
        for item in z.infolist():
            if item.is_dir():
                continue
            relative = Path(item.filename.removeprefix(prefix))
            assert ".." not in relative.parts and not relative.is_absolute()
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
    split = {}
    for line in (stage / "train_test_split.yaml").read_text().splitlines():
        key, value = line.split(":", 1)
        split[key] = ast.literal_eval(value.strip())
    train, test = set(split["train"]), set(split["test"])
    overlap = sorted(train & test)
    reference_pool = sorted(train - test)
    targets = sorted(test)
    assert len(train | test) == 60
    assert len(reference_pool) == 39 and len(targets) == 21 and overlap == [28]
    assert set(reference_pool).isdisjoint(targets)
    normalized = stage / "vegetation_masks"
    normalized.mkdir()
    images = sorted((stage / "images").glob("*_image.png"))
    assert len(images) == len(list((stage / "masks").glob("*_mask.png"))) == 60
    for image in images:
        identity = image.stem.split("_")[0]
        with Image.open(image) as im:
            im.load()
            size = im.size
        with Image.open(stage / "masks" / (identity + "_mask.png")) as mask:
            assert mask.size == size
            values = np.asarray(mask.convert("L"))
            assert set(np.unique(values).tolist()) == {0, 255}
            # Author's binary masks use black for vegetation, white for soil.
            vegetation = values < 128
        Image.fromarray(vegetation.astype(np.uint8) * 255).save(normalized / (identity + ".png"))
    protocol = {
        "dataset": "CWFID", "source": "https://github.com/cwfid/dataset",
        "archive_sha256": EXPECTED_SHA, "task": "all vegetation foreground segmentation",
        "mask_conversion": "author binary mask: 0=vegetation, 255=background; normalized to 255=foreground",
        "original_split_overlap": overlap, "resolution": "native images; official INSID3 model input=1024",
        "reference_pool": reference_pool, "test_ids": targets,
        "reference_id": min(reference_pool),
        "reference_selection": "lowest ID in author train split after excluding all author test IDs; fixed before inference",
        "shots": 1, "finetuning": False, "test_ground_truth_used": "metrics only; never passed to model",
        "license": "non-commercial research; cite Haug and Ostermann, ECCV 2014 Workshops",
    }
    (stage / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    stage.rename(final)
    report = dict(protocol, status="PASS", images=60, masks=60, test_images=21,
                  dataset_directory=str(final))
    (ROOT.parent / "setup-logs/cwfid-data-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print("CWFID_DATA_CHECK_PASS", final)


if __name__ == "__main__":
    main()
