"""Evaluate frozen INSID3 on CWFID vegetation and estimate vegetation cover."""
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


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MPLBACKEND"] = "Agg"
    os.chdir(ROOT)
    import numpy as np
    import torch
    from PIL import Image
    import matplotlib.pyplot as plt
    import models
    from reproduce import CONFIG, WEIGHTS

    data = ROOT / "data/CWFID"
    protocol = json.loads((data / "protocol.json").read_text())
    assert protocol["reference_id"] not in protocol["test_ids"]
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output = ROOT / "output" / ("cwfid_" + datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir()
    print("Output directory:", output, flush=True)
    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    configuration = dict(configuration=CONFIG, protocol=protocol, physical_gpu=args.gpu,
                         gpu_name=torch.cuda.get_device_name(0), torch=torch.__version__,
                         weights_sha256=digest(WEIGHTS), entrypoint_sha256=digest(__file__),
                         official_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    assert configuration["weights_sha256"] == "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"
    write("configuration.json", configuration)
    local = Path(torch.hub.get_dir()) / "facebookresearch_dinov3_main"
    models._build_encoder = lambda model_size="large": torch.hub.load(
        str(local), models._HUB_NAMES[model_size], source="local", weights=str(ROOT / models._WEIGHTS[model_size]))
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    records = []
    total_inter = total_union = total_pred = total_gt = 0
    ref_id = protocol["reference_id"]
    reference = str(data / "images" / f"{ref_id:03d}_image.png")
    reference_mask = str(data / "vegetation_masks" / f"{ref_id:03d}.png")
    with torch.inference_mode():
        model = models.build_insid3(model_size="large", image_size=1024, svd_components=500,
                                    tau=.6, merge_threshold=.2, mask_refiner="crf", crf_size=640, device="cuda")
        model.eval()
        for index, identity in enumerate(protocol["test_ids"]):
            image = data / "images" / f"{identity:03d}_image.png"
            model.set_reference(reference, reference_mask)
            model.set_target(str(image))
            prediction = model.segment().cpu().numpy().astype(bool)
            # Read target annotation only after inference; it is not a model input.
            truth = np.asarray(Image.open(data / "vegetation_masks" / f"{identity:03d}.png").convert("L")) >= 128
            assert prediction.shape == truth.shape
            inter, union = int((prediction & truth).sum()), int((prediction | truth).sum())
            n_pred, n_gt = int(prediction.sum()), int(truth.sum())
            record = dict(target_id=identity, reference_id=ref_id, iou_percent=inter/union*100,
                          dice_percent=2*inter/(n_pred+n_gt)*100,
                          predicted_cover_percent=prediction.mean()*100, true_cover_percent=truth.mean()*100,
                          intersection=inter, union=union)
            record["cover_absolute_error_pp"] = abs(record["predicted_cover_percent"]-record["true_cover_percent"])
            records.append(record)
            total_inter += inter
            total_union += union
            total_pred += n_pred
            total_gt += n_gt
            Image.fromarray(prediction.astype(np.uint8)*255).save(output / f"{identity:03d}_predicted_mask.png")
            if index < 3:
                fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
                for ax, visual, title in zip(axes, [Image.open(image), truth, prediction], ["Target", "Ground truth: vegetation", "INSID3 prediction"]):
                    ax.imshow(visual, cmap="gray")
                    ax.set_title(title)
                    ax.axis("off")
                fig.suptitle(f"CWFID target {identity:03d}, fixed reference {ref_id:03d} | IoU {record['iou_percent']:.2f}%")
                fig.savefig(output / f"{identity:03d}_comparison.png", dpi=130)
                plt.close(fig)
            print(f"{index+1}/{len(protocol['test_ids'])} target={identity:03d} IoU={record['iou_percent']:.2f}%", flush=True)
    with (output / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    torch.cuda.synchronize()
    result = dict(status="completed", dataset="CWFID", task="vegetation foreground + cover estimation",
                  test_images=len(records), reference_id=ref_id, shots=1, finetuning=False,
                  foreground_iou_percent=total_inter/total_union*100,
                  mean_per_image_iou_percent=float(np.mean([r["iou_percent"] for r in records])),
                  pooled_dice_percent=2*total_inter/(total_pred+total_gt)*100,
                  mean_per_image_dice_percent=float(np.mean([r["dice_percent"] for r in records])),
                  cover_mae_percentage_points=float(np.mean([r["cover_absolute_error_pp"] for r in records])),
                  total_seconds=time.perf_counter()-started, peak_gpu_memory_mib=torch.cuda.max_memory_allocated()/1024**2,
                  paper_comparison=None, scope="independent application protocol; not an INSID3 paper benchmark")
    write("result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
