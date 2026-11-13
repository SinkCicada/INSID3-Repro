"""Run official INSID3 demos, SUIM evaluation, and chest X-ray application."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "pretrain/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
CONFIG = dict(model_size="large", image_size=1024, shots=1, seed=0,
              mask_refiner="crf", crf_size=640, svd_components=500,
              tau=0.6, merge_threshold=0.2)
PAPER_SUIM_MIOU = 54.9
PAPER_LUNG_MIOU = 78.8


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "demo", "suim", "lung", "isic", "isaid"])
    parser.add_argument("--fold", type=int, choices=[0,1,2], help="Required for iSAID; evaluate every fold before comparing with the paper.")
    parser.add_argument("--gpu", type=int, help="Physical GPU index; required for demo/evaluation.")
    parser.add_argument("--reference-image", type=Path, default=ROOT / "assets/ref_cat_image.jpg")
    parser.add_argument("--reference-mask", type=Path, default=ROOT / "assets/ref_cat_mask.png")
    parser.add_argument("--target-image", type=Path, default=ROOT / "assets/target_cat_image.jpg")
    args = parser.parse_args()
    if args.mode == "isaid" and args.fold is None:
        parser.error("iSAID requires an explicit --fold 0, 1 or 2.")
    if args.mode != "isaid" and args.fold is not None:
        parser.error("--fold is currently implemented only for iSAID.")
    args.reference_image = args.reference_image.resolve()
    args.reference_mask = args.reference_mask.resolve()
    args.target_image = args.target_image.resolve()

    if args.mode == "check":
        report_path = ROOT.parent / "setup-logs/suim-data-check.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        print(json.dumps({
            "weights_present": WEIGHTS.is_file(), "weights_path": str(WEIGHTS),
            "suim_data_check": report, "paper_config": CONFIG,
            "isic_data_present": (ROOT / "data/ISIC/ISIC2018_Task1-2_Training_Input/1").is_dir(),
            "lung_data_present": (ROOT / "data/LungSegmentation/CXR_png").is_dir(),
            "demo_inputs_present": all(p.is_file() for p in
                                      [args.reference_image, args.reference_mask, args.target_image])
        }, indent=2))
        return

    if not WEIGHTS.is_file():
        parser.error("Missing official Large weights: " + str(WEIGHTS))
    if args.gpu is None or args.gpu < 0:
        parser.error("Select an available GPU explicitly, for example --gpu 1.")
    if args.mode == "demo":
        for path in [args.reference_image, args.reference_mask, args.target_image]:
            if not path.is_file():
                parser.error("Missing demo input: " + str(path))
    if args.mode in ("suim", "lung", "isic", "isaid"):
        folder = {"suim": "SUIM/images", "lung": "LungSegmentation/CXR_png", "isic": "ISIC/ISIC2018_Task1-2_Training_Input/1", "isaid": "iSAID/val/images"}[args.mode]
        if not (ROOT / "data" / folder).is_dir():
            parser.error("Missing prepared dataset: " + folder)

    weight_sha256 = digest(WEIGHTS)
    if not weight_sha256.startswith("8aa4cbdd"):
        parser.error("The checkpoint SHA-256 does not match the official filename prefix.")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(ROOT)

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable in the selected Python environment/GPU.")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_id = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / "output" / (args.mode + (("_fold" + str(args.fold)) if args.mode == "isaid" else "") + "_" + run_id)
    output.mkdir(parents=True)
    manifest = {
        "mode": args.mode, "configuration": CONFIG, "fold": args.fold,
        "started_at": datetime.now(timezone(timedelta(hours=8))).isoformat(), "physical_gpu": args.gpu,
        "gpu_name": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "weights_sha256": weight_sha256,
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "entrypoint_sha256": digest(Path(__file__)),
        "execution": "torch.inference_mode; official model and metric implementation",
        "output_directory": str(output),
    }
    save_json(output / "configuration.json", manifest)
    print("Output directory:", output, flush=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    # Load the already downloaded official DINOv3 source without a GitHub probe.
    import models
    local_backbone = Path(torch.hub.get_dir()) / "facebookresearch_dinov3_main"
    if not (local_backbone / "hubconf.py").is_file():
        parser.error("Missing cached official DINOv3 source: " + str(local_backbone))
    def build_local_encoder(model_size="large"):
        return torch.hub.load(str(local_backbone), models._HUB_NAMES[model_size],
                              source="local", weights=str(ROOT / models._WEIGHTS[model_size]))
    models._build_encoder = build_local_encoder
    print("Loading official DINOv3 from local source and weights.", flush=True)

    with torch.inference_mode():
        if args.mode == "demo":
            from PIL import Image
            from models import build_insid3
            from utils.visualization import visualize_prediction_segmentation

            model = build_insid3(model_size="large", image_size=1024,
                                 svd_components=500, tau=0.6, merge_threshold=0.2,
                                 mask_refiner="crf", crf_size=640, device="cuda")
            model.eval()
            torch.cuda.synchronize()
            initialized_seconds = time.perf_counter() - started
            reference = str(args.reference_image.resolve())
            reference_mask = str(args.reference_mask.resolve())
            target = str(args.target_image.resolve())
            model.set_reference(reference, reference_mask)
            model.set_target(target)
            mask = model.segment()
            Image.fromarray(mask.cpu().numpy().astype(np.uint8) * 255).save(
                output / "predicted_mask.png")
            visualize_prediction_segmentation(
                reference, reference_mask, target, mask, output / "prediction.png")
            result = {
                "status": "completed", "mode": "demo",
                "initialization_seconds": initialized_seconds,
                "reference_image": reference, "reference_mask": reference_mask,
                "target_image": target, "predicted_mask_shape": list(mask.shape),
            }
        else:
            import opts
            from inference_segmentation import main as official_main

            dataset_name = {"suim": "SUIM", "lung": "Chest X-Ray", "isic": "ISIC", "isaid": "iSAID"}[args.mode]
            paper_miou = {"suim": PAPER_SUIM_MIOU, "lung": PAPER_LUNG_MIOU, "isic": 54.4, "isaid": 52.1}[args.mode]
            official_args = opts.get_args_parser().parse_args([
                "--dataset", args.mode, "--model-size", "large",
                "--image-size", "1024", "--shots", "1",
                "--svd-comps", "500", "--tau", "0.6", "--merge-thresh", "0.2",
                "--crf-mask-refinement", "--crf-size", "640",
                "--seed", "0", "--num-workers", "0", "--device", "cuda",
                "--data-root", str(ROOT / "data"),
                "--output-dir", str(output), "--exp-name", "insid3-" + args.mode,
                *(["--fold", str(args.fold)] if args.mode == "isaid" else []),
            ])
            save_json(output / "official_arguments.json", vars(official_args))
            miou = float(official_main(official_args))
            result = {
                "status": "completed", "dataset": dataset_name, "shots": 1,
                "miou_percent": miou, "paper_miou_percent": paper_miou if args.mode != "isaid" else None,
                "difference_percentage_points": miou - paper_miou if args.mode != "isaid" else None,
                "paper_reference": "INSID3 Table 1; " + dataset_name + "; 1-shot",
            }
            if args.mode == "isaid":
                result.update(fold=args.fold, paper_aggregate_miou_percent=52.1,
                              comparison_scope="single fold only; paper comparison requires all three folds")
            with (output / "comparison.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["dataset", "paper_mIoU", "reproduced_mIoU", "difference_pp"])
                writer.writerow([dataset_name + ((" fold " + str(args.fold)) if args.mode == "isaid" else ""), result["paper_miou_percent"], miou, result["difference_percentage_points"]])

    torch.cuda.synchronize()
    result["total_seconds"] = time.perf_counter() - started
    result["peak_gpu_memory_mib"] = torch.cuda.max_memory_allocated() / 1024**2
    save_json(output / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
