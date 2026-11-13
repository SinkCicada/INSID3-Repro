"""Evaluate one official PACO-Part or LVIS fold and preserve per-class IoUs."""
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
    parser.add_argument("--dataset", choices=["paco", "lvis"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    limit = 4 if args.dataset == "paco" else 10
    if not 0 <= args.fold < limit:
        parser.error("Fold outside the official range.")
    dataset_key = "paco_part" if args.dataset == "paco" else "lvis"
    required = ROOT.parent / ("setup-logs/" + args.dataset + "-data-check.json")
    if not required.is_file():
        parser.error("Missing validated complete dataset report: " + str(required))
    if args.gpu < 0:
        parser.error("Select a physical GPU index >= 0.")
    if not WEIGHTS.is_file():
        parser.error("Missing official DINOv3 Large weights.")
    import json
    report = json.loads(required.read_text())
    if report.get("status") != "PASS":
        parser.error("Data verification did not pass.")
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
    import inference_segmentation as official
    if args.dataset == "paco":
        from datasets.paco_part import DatasetPACOPart as DatasetClass
    else:
        from datasets.lvis import DatasetLVIS as DatasetClass
    captured = []
    OriginalMeter = official.AverageMeter
    class CapturingMeter(OriginalMeter):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            captured.append(self)
    official.AverageMeter = CapturingMeter
    dataset = DatasetClass(str(ROOT / "data"), fold=args.fold, shot=1)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    output = ROOT / "output" / (f"{args.dataset}_fold{args.fold}_" + datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir()
    save_json(output / "configuration.json", {
        "mode": dataset_key, "fold": args.fold, "configuration": CONFIG,
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
        "--dataset", dataset_key, "--fold", str(args.fold), "--model-size", "large",
        "--image-size", "1024", "--shots", "1", "--svd-comps", "500",
        "--tau", "0.6", "--merge-thresh", "0.2", "--crf-mask-refinement",
        "--crf-size", "640", "--seed", "0", "--num-workers", "0", "--device", "cuda",
        "--data-root", str(ROOT / "data"), "--output-dir", str(output),
        "--exp-name", "insid3-" + args.dataset,
    ])
    save_json(output / "official_arguments.json", vars(official_args))
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        miou = float(official.main(official_args))
    torch.cuda.synchronize()
    assert len(captured) == 1
    meter = captured[0]
    intersection = meter.intersection_buf[1, dataset.class_ids].detach().cpu().tolist()
    union = meter.union_buf[1, dataset.class_ids].detach().cpu().tolist()
    per_class = [{"class_id": cid, "original_class_id": int(name), "intersection": inter, "union": total,
                  "iou_percent": inter / max(total, 1) * 100}
                 for cid, name, inter, total in zip(dataset.class_ids, dataset.class_ids_ori, intersection, union)]
    assert abs(sum(c["iou_percent"] for c in per_class)/len(per_class)-miou) < 0.001
    save_json(output / "per_class_iou.json", per_class)
    result = {
        "status": "completed", "dataset": "PACO-Part" if args.dataset == "paco" else "LVIS-92i", "fold": args.fold, "shots": 1,
        "miou_percent": miou, "paper_miou_percent": None,
        "paper_aggregate_miou_percent": 38.7 if args.dataset == "paco" else 41.8, "difference_percentage_points": None,
        "comparison_scope": "one official fold; report all folds before paper comparison",
        "dataset_key": dataset_key, "class_count": len(dataset.class_ids), "episodes": len(dataset),
        "per_class_iou": per_class, "box_crop": getattr(dataset, "box_crop", False),
        "total_seconds": time.perf_counter()-started,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated()/1024**2,
    }
    save_json(output / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
