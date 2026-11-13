from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import shutil
import zipfile
import numpy as np
import torch
from datasets import build_dataset

project = Path(__file__).resolve().parent
root = project.parent
archive = root / "downloads/chest-xray-masks-and-labels.zip"
final = project / "data/LungSegmentation"
staging_parent = root / "downloads/lung-prepared"
staging = staging_parent / "LungSegmentation"
assert not final.exists(), final
assert zipfile.is_zipfile(archive), archive
staging.mkdir(parents=True, exist_ok=True)
prefix = "Lung Segmentation/"
with zipfile.ZipFile(archive) as z:
    bad = z.testzip()
    assert bad is None, bad
    members = [i for i in z.infolist() if not i.is_dir()
               and (i.filename.startswith(prefix + "CXR_png/")
                    or i.filename.startswith(prefix + "masks/"))]
    assert members, "Expected folders absent; inspect archive layout."
    for info in members:
        relative = Path(info.filename[len(prefix):])
        destination = staging / relative
        assert destination.resolve().is_relative_to(staging.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as source, destination.open("wb") as output:
            shutil.copyfileobj(source, output)

images = list((staging / "CXR_png").glob("*.png"))
masks = list((staging / "masks").glob("*.png"))
def image_name(mask):
    name = Path(mask).name
    return name if "MCUCXR" in name else name[:-9] + ".png"
missing = [str(p) for p in masks if not (staging / "CXR_png" / image_name(p)).is_file()]
assert images and len(masks) >= 2 and not missing, missing[:5]
torch.set_num_threads(2)
np.random.seed(0)
ds = build_dataset("lung", SimpleNamespace(data_root=str(staging_parent), shots=1, fold=0))
for i in range(3):
    batch = ds[i]
    assert len(batch["ref_imgs"]) == len(batch["ref_masks"]) == 1
    assert batch["tgt_mask"].ndim == 2
np.random.seed(0)
target, references, class_id = ds.sample_episode(0)
ref = Path(references[0])
target = Path(target)
case = {
    "reference_image": str(Path("data/LungSegmentation/CXR_png") / image_name(ref)),
    "reference_mask": str(Path("data/LungSegmentation/masks") / ref.name),
    "target_image": str(Path("data/LungSegmentation/CXR_png") / image_name(target)),
    "target_ground_truth_mask": str(Path("data/LungSegmentation/masks") / target.name),
}
staging.rename(final)
assert all((project / value).is_file() for value in case.values())
(project / "examples").mkdir(exist_ok=True)
(project / "examples/lung-demo-case.json").write_text(json.dumps(case, indent=2) + "\n")
h = hashlib.sha256()
with archive.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        h.update(chunk)
report = {
    "status": "PASS", "dataset_path": str(final),
    "source_url": "https://www.kaggle.com/api/v1/datasets/download/nikhilpandey360/chest-xray-masks-and-labels",
    "images": len(images), "masks": len(masks),
    "evaluation_episodes_per_run": len(ds), "sampled_episode_checks": 3,
    "archive_sha256": h.hexdigest(),
}
archive.rename(archive.with_suffix(""))
(root / "setup-logs/lung-data-check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
print("LUNG_DATA_CHECK_PASS", flush=True)
