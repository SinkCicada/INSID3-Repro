"""Evaluate the released ISIC protocol and save its sampled episode IDs."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

import inference_segmentation as official
import models
import opts
from datasets.isic import DatasetISIC

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--paired-grids", action="store_true")
options = parser.parse_args()
options.output.mkdir()
args = opts.get_args_parser().parse_args([
    "--dataset", "isic", "--data-root", str(Path("data").resolve()),
    "--model-size", "large", "--image-size", "1024", "--shots", "1",
    "--svd-comps", "500", "--tau", "0.6", "--merge-thresh", "0.2",
    "--crf-mask-refinement", "--crf-size", "640", "--seed", str(options.seed),
    "--num-workers", "0", "--device", "cuda",
    "--output-dir", str(options.output), "--exp-name", "isic-seed-sensitivity",
])
local_backbone = Path(torch.hub.get_dir()) / "facebookresearch_dinov3_main"
assert (local_backbone / "hubconf.py").is_file()
models._build_encoder = lambda model_size: torch.hub.load(
    str(local_backbone), models._HUB_NAMES[model_size],
    source="local", weights=models._WEIGHTS[model_size],
)

def save(name, value):
    path = options.output / name
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

save("official_arguments.json", vars(args))
episode_ids = []
sample_episode = DatasetISIC.sample_episode

def capture_episode(self, index):
    query, refs, class_id = sample_episode(self, index)
    episode_ids.append({
        "episode": index, "class_id": class_id,
        "query": Path(query).stem, "support": Path(refs[0]).stem,
    })
    return query, refs, class_id

DatasetISIC.sample_episode = capture_episode
if options.paired_grids:
    from paired_grids import compare_grids
    assert args.seed == 0
    official.evaluate = compare_grids
start = time.perf_counter()
with torch.inference_mode():
    miou = official.main(args)
assert len(episode_ids) == 600
assert [sum(e["class_id"] == c for e in episode_ids) for c in range(3)] == [200] * 3
save("episodes.json", episode_ids)
if options.paired_grids:
    historical = json.loads(Path("results/table1.json").read_text())
    reference = next(row["miou_percent"] for row in historical if row["dataset"] == "ISIC")
    save("grid_comparison.json", {
        "seed": args.seed, "episodes": len(episode_ids),
        "historical_seed0_miou_percent": reference,
        "baseline_matches_within_0_01_pp": abs(miou - reference) <= 0.01,
        "results": compare_grids.results,
    })
save("result.json", {
    "seed": args.seed, "episodes": len(episode_ids), "miou_percent": miou,
    "paper_miou_percent": 54.4, "elapsed_seconds": time.perf_counter() - start,
    "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "gpu_name": torch.cuda.get_device_name(), "torch_version": torch.__version__,
})
