"""Evaluate one official COCO-20i fold without changing active iSAID entrypoints."""
import argparse
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import random
import subprocess
import time

from reproduce import CONFIG, WEIGHTS, digest, save_json

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    required = ROOT / "data/COCO2014/splits/val" / f"fold{args.fold}.pkl"
    if not required.is_file():
        parser.error("Missing prepared COCO fold: " + str(required))
    if args.gpu < 0:
        parser.error("Select a physical GPU index >= 0.")
    if not WEIGHTS.is_file():
        parser.error("Missing official DINOv3 Large weights.")
    weight_sha = digest(WEIGHTS)
    if weight_sha != "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035":
        parser.error("The checkpoint differs from the verified official Large weights.")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MPLBACKEND"] = "Agg"
    os.chdir(ROOT)
    import numpy as np
    import torch
    import models
    import opts
    from inference_segmentation import main as official_main

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    output = ROOT / "output" / (f"coco_fold{args.fold}_" + datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir()
    save_json(output / "configuration.json", {
        "mode": "coco", "fold": args.fold, "configuration": CONFIG,
        "physical_gpu": args.gpu, "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "weights_sha256": weight_sha,
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "entrypoint_sha256": digest(__file__), "base_helper_sha256": digest(ROOT / "reproduce.py"),
        "data_report_sha256": digest(ROOT.parent / "setup-logs/coco-data-check.json"),
        "execution": "torch.inference_mode; official model, sampling and metrics",
        "output_directory": str(output),
    })
    print("Output directory:", output, flush=True)
    cached = Path(torch.hub.get_dir()) / "facebookresearch_dinov3_main"
    if not (cached / "hubconf.py").is_file():
        parser.error("Missing cached official DINOv3 source.")
    models._build_encoder = lambda model_size="large": torch.hub.load(
        str(cached), models._HUB_NAMES[model_size], source="local",
        weights=str(ROOT / models._WEIGHTS[model_size]))
    official_args = opts.get_args_parser().parse_args([
        "--dataset", "coco", "--fold", str(args.fold), "--model-size", "large",
        "--image-size", "1024", "--shots", "1", "--svd-comps", "500",
        "--tau", "0.6", "--merge-thresh", "0.2", "--crf-mask-refinement",
        "--crf-size", "640", "--seed", "0", "--num-workers", "0", "--device", "cuda",
        "--data-root", str(ROOT / "data"), "--output-dir", str(output),
        "--exp-name", "insid3-coco",
    ])
    save_json(output / "official_arguments.json", vars(official_args))
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        miou = float(official_main(official_args))
    torch.cuda.synchronize()
    result = {
        "status": "completed", "dataset": "COCO-20i", "fold": args.fold, "shots": 1,
        "miou_percent": miou, "paper_miou_percent": None,
        "paper_aggregate_miou_percent": 57.6, "difference_percentage_points": None,
        "comparison_scope": "one fold only; paper comparison requires all four folds",
        "total_seconds": time.perf_counter()-started,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated()/1024**2,
    }
    save_json(output / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
