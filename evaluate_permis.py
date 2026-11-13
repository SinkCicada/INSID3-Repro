"""Evaluate the complete official PerMIS 1-shot benchmark."""
import argparse
import json
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
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    required = ROOT.parent / "setup-logs/permis-data-check.json"
    if not required.is_file():
        parser.error("Missing verified PerMIS data report: " + str(required))
    report = json.loads(required.read_text())
    if report.get("status") != "PASS" or report.get("cases") != 216 or report.get("episodes") != 432:
        parser.error("Complete PerMIS data verification is required.")
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
    output = ROOT / "output" / ("permis_" + datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir()
    save_json(output / "configuration.json", {
        "mode": "permis", "configuration": CONFIG,
        "physical_gpu": args.gpu, "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "weights_sha256": weight_sha,
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "entrypoint_sha256": digest(__file__), "base_helper_sha256": digest(ROOT / "reproduce.py"),
        "data_report_sha256": digest(required),
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
        "--dataset", "permis", "--fold", "0", "--model-size", "large",
        "--image-size", "1024", "--shots", "1", "--svd-comps", "500",
        "--tau", "0.6", "--merge-thresh", "0.2", "--crf-mask-refinement",
        "--crf-size", "640", "--seed", "0", "--num-workers", "0", "--device", "cuda",
        "--data-root", str(ROOT / "data"), "--output-dir", str(output),
        "--exp-name", "insid3-permis",
    ])
    save_json(output / "official_arguments.json", vars(official_args))
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        miou = float(official_main(official_args))
    torch.cuda.synchronize()
    result = {
        "status": "completed", "dataset": "PerMIS", "shots": 1,
        "miou_percent": miou, "paper_miou_percent": 67.0,
        "difference_percentage_points": miou - 67.0,
        "comparison_scope": "all 216 cases, 432 queries",
        "official_metric_scope": "DatasetPerMis class_ids=[0]; original evaluator accumulates foreground intersection/union over all queries",
        "cases": 216, "episodes": 432, "paper_reference": "INSID3 Table 1; PerMIS; 1-shot",
        "total_seconds": time.perf_counter()-started,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated()/1024**2,
    }
    save_json(output / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
