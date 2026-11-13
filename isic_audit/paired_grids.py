"""Score several CRF and mask grids on the same 1-shot ISIC episodes."""
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import inference_segmentation as official
from utils.refinement import crf_refine, init_crf, upsample_mask


def compare_grids(args, model, log_file):
    dataset = official.build_dataset(args.dataset, args=args)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=lambda batch: batch[0],
    )
    meters = {
        label: official.AverageMeter(args.dataset, dataset.class_ids, device=args.device)
        for label in ("crf640_score1024", "crf1024_score1024", "crf640_score512",
                      "crf512_score512", "crf640_score_native_mask")
    }
    crfs = {
        640: (model._crf, model._crf_band_px, model._crf_p_core),
        1024: init_crf(1024, args.device),
        512: init_crf(512, args.device),
    }
    refinements = {}

    def finish_mask(patch_mask, target):
        for grid, (crf, band, p_core) in crfs.items():
            image = target
            if target.shape[-2:] != (grid, grid):
                image = F.interpolate(target, size=(grid, grid), mode="bilinear",
                                      align_corners=False)
            refinements[grid] = crf_refine(
                crf, band, p_core, image, upsample_mask(patch_mask, grid, grid)
            )
        return upsample_mask(refinements[640], 1024, 1024)

    model._finalize_mask = finish_mask
    started = time.perf_counter()
    for index, batch in enumerate(loader, start=1):
        model.set_reference(batch["ref_imgs"][0], batch["ref_masks"][0])
        model.set_target(batch["tgt_img"])
        pred = model.segment()
        gt = batch["tgt_mask"].to(args.device)
        native_shape = gt.shape
        gt_1024 = F.interpolate(gt[None, None], size=(1024, 1024), mode="nearest")[0, 0]
        gt_512 = F.interpolate(gt[None, None], size=(512, 512), mode="nearest")[0, 0]
        variants = {
            "crf640_score1024": (pred, gt_1024),
            "crf1024_score1024": (refinements[1024], gt_1024),
            "crf640_score512": (upsample_mask(refinements[640], 512, 512), gt_512),
            "crf512_score512": (refinements[512], gt_512),
            "crf640_score_native_mask": (upsample_mask(refinements[640], *native_shape), gt),
        }
        for label, (mask, truth) in variants.items():
            intersection, union = official.Evaluator.classify_prediction(mask, truth)
            meters[label].update(intersection, union, batch["class_id"].to(args.device))
        if index == 1 or index % 25 == 0:
            elapsed = time.perf_counter() - started
            print(f"Paired grids {index}/{len(dataset)} episodes, {elapsed:.1f}s", flush=True)

    results = {}
    for label, meter in meters.items():
        intersections = meter.intersection_buf[1].cpu().tolist()
        unions = meter.union_buf[1].cpu().tolist()
        results[label] = {
            "miou_percent": meter.compute_iou()[0].item(),
            "per_class_iou_percent": [100 * i / max(1, u) for i, u in zip(intersections, unions)],
            "intersections": intersections,
            "unions": unions,
        }
    compare_grids.results = results
    baseline = results["crf640_score1024"]["miou_percent"]
    with open(log_file, "a", encoding="utf-8") as stream:
        for label, result in results.items():
            stream.write(f"{label}: {result['miou_percent']:.8f}\n")
    return baseline
