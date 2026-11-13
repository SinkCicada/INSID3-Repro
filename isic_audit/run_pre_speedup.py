"""Evaluate the released ISIC protocol and save its sampled episode IDs."""
import argparse
import json
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
options = parser.parse_args()
options.output.mkdir()
args = opts.get_args_parser().parse_args([
    "--dataset", "isic", "--data-root", str(Path("data").resolve()),
    "--model-size", "large", "--image-size", "1024", "--shots", "1",
    "--svd-comps", "500", "--tau", "0.6", "--merge-thresh", "0.2",
    "--crf-mask-refinement", "--seed", str(options.seed),
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
start = time.perf_counter()
with torch.inference_mode():
    miou = official.main(args)
assert len(episode_ids) == 600
assert [sum(e["class_id"] == c for e in episode_ids) for c in range(3)] == [200] * 3
save("episodes.json", episode_ids)
save("result.json", {
    "seed": args.seed, "episodes": len(episode_ids), "miou_percent": miou,
    "paper_miou_percent": 54.4, "elapsed_seconds": time.perf_counter() - start,
    "official_commit": "26759bd9a98490b43882e51a32e5e8a3a6605086",
    "gpu_name": torch.cuda.get_device_name(), "torch_version": torch.__version__,
    "crf_resolution": [1024, 1024], "score_resolution": [1024, 1024],
    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
})
