"""Run a complete released-data fold with 5 references or a controlled ablation.

The official dataset, seed setup, evaluator, and model remain unchanged.
Ablations modify only the documented expression in an in-memory method copy.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import textwrap
import time
from datetime import datetime, timezone

from extended_experiment_spec import ROOT, LOGS, FOLDS, VARIANTS, arguments


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def predict_threshold(self, ref_images, ref_masks, tgt_image):
    """Table 3 no-clustering baseline, using the mean prototype in Eq. (2)."""
    import torch
    import torch.nn.functional as F
    from utils.data import downsample_mask

    assert ref_images.shape[0] == ref_masks.shape[0] == 1
    images = torch.cat((ref_images, tgt_image), dim=0).unsqueeze(0)
    reference, target = F.normalize(self._extract_features(images), p=2, dim=2)[0]
    height, width = reference.shape[-2:]
    if not ref_masks.any():
        mask = torch.zeros((height, width), dtype=torch.bool, device=target.device)
    else:
        foreground = downsample_mask(ref_masks.unsqueeze(1), height, width)
        prototype = reference[:, foreground].mean(dim=1)
        similarity = torch.einsum('chw,c->hw', target, prototype)
        mask = similarity > 0.55
    return self._finalize_mask(mask, tgt_image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=FOLDS, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--shots', type=int, choices=[1,5], required=True)
    parser.add_argument('--variant', choices=['full','threshold_055']+VARIANTS, default='full')
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    assert 0 <= args.fold < FOLDS[args.dataset] and args.gpu >= 0
    assert args.variant == 'full' or (args.shots == 1 and args.dataset in ['coco','pascal_part'])
    report = LOGS / 'five-shot-data-check.json'
    checked = json.loads(report.read_text())
    assert checked['status'] == 'PASS'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['MPLBACKEND'] = 'Agg'
    os.chdir(ROOT)
    # This host's THP allocation made numpy.triu_indices(4096) take 80 s.
    # Disable THP only for this process, before importing scientific libraries.
    # Verified against /usr/include/linux/prctl.h: PR_SET_THP_DISABLE == 41.
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(41, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot disable THP for this process')
    import torch
    import opts
    import models
    import inference_segmentation as official
    import models.insid3 as model_module

    weights = ROOT / models._WEIGHTS['large']
    weight_sha = digest(weights)
    assert weight_sha == '8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035'
    output = ROOT / 'output' / (f'{args.dataset}_{args.shots}shot_{args.variant}_fold{args.fold}_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir()
    print('Output directory:', output, flush=True)
    official_args = opts.get_args_parser().parse_args(arguments(args.dataset,args.fold,args.shots,output,args.variant))
    cached = Path(torch.hub.get_dir()) / 'facebookresearch_dinov3_main'
    assert (cached / 'hubconf.py').is_file()
    models._build_encoder = lambda model_size='large': torch.hub.load(
        str(cached), models._HUB_NAMES[model_size], source='local',
        weights=str(ROOT / models._WEIGHTS[model_size]))

    # Assert the exact frozen source expression before constructing any overlay.
    description = 'Unmodified released INSID3 model'
    if args.variant in ['seed_only_coarse','seed_only_fine','cross_only']:
        original = textwrap.dedent(inspect.getsource(model_module.INSID3._seed_and_aggregate))
        if args.variant.startswith('seed_only'):
            old = 'seed_id = matched_ids[seed_idx].item()'
            new = old+'\n    return cluster_labels == seed_id'
            description = 'Keep the original candidate localization and seed selection; return only the selected seed cluster'
        else:
            old = 'combined = cross_sim * intra_sim'
            new = 'combined = cross_sim.clone()'
            description = 'Replace cross_sim * intra_sim with cross_sim; preserve original area weighting and threshold'
        assert original.count(old) == 1
        overlay = original.replace(old,new)
        (output/'ablation_overlay.py').write_text(overlay)
        namespace = {}
        exec(compile(overlay,'<controlled-ablation>','exec'),vars(model_module),namespace)
        model_module.INSID3._seed_and_aggregate = namespace['_seed_and_aggregate']
    elif args.variant == 'no_debias':
        description = 'Official --svd-comps 0: the positional basis is empty and the projection is the identity'
    elif args.variant == 'threshold_055':
        description = ('Table 3 Eq. (2) reconstruction: unit-normalized original patch features, '
                       'unrenormalized mean foreground prototype, similarity > 0.55, '
                       'same released mask downsampling and CRF as the main benchmark. '
                       'Feature normalization and CRF are controlled choices in the absence of a released baseline implementation.')
        (output/'ablation_overlay.py').write_text(inspect.getsource(predict_threshold))
        model_module.INSID3.predict_mask = predict_threshold

    capture = {'datasets':[], 'meters':[], 'episodes':0, 'references':{}}
    original_builder = official.build_dataset
    original_meter = official.AverageMeter

    class AuditedDataset:
        def __init__(self, dataset):
            self.dataset = dataset
            self.class_ids = dataset.class_ids
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, index):
            batch = self.dataset[index]
            nref = len(batch['ref_imgs'])
            assert nref == args.shots == len(batch['ref_masks'])
            assert all(image.width > 0 and image.height > 0 for image in batch['ref_imgs'])
            capture['episodes'] += 1
            capture['references'][nref] = capture['references'].get(nref,0)+1
            return batch

    def audited_builder(name, args):
        dataset = original_builder(name,args=args)
        capture['datasets'].append(dataset)
        return AuditedDataset(dataset)

    class CapturedMeter(original_meter):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            capture['meters'].append(self)

    official.build_dataset = audited_builder
    official.AverageMeter = CapturedMeter
    sources = ['inference_segmentation.py','models/insid3.py','opts.py',
               'utils/metrics.py','utils/data.py','datasets/'+args.dataset+'.py',
               'extended_experiment_spec.py','evaluate_extended_segmentation.py']
    save(output/'configuration.json', {
        'arguments':vars(official_args),'physical_gpu':args.gpu,
        'weights_sha256':weight_sha,'data_report_sha256':digest(report),
        'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_sha256':{p:digest(ROOT/p) for p in sources},
        'variant':args.variant,'variant_definition':description,
        'runtime_memory_policy':'PR_SET_THP_DISABLE=1 for this process only; no system-wide change',
        'execution':'official main/evaluate under torch.inference_mode; sampling unchanged; per-episode reference-count assertions',
        'paper_comparison_note':'Released-code reproduction; unresolved paper class filtering/aggregation is retained explicitly.',
    })
    started = time.perf_counter()
    with torch.inference_mode():
        miou = float(official.main(official_args))
    torch.cuda.synchronize()
    assert len(capture['datasets']) == len(capture['meters']) == 1
    dataset, meter = capture['datasets'][0], capture['meters'][0]
    assert capture['episodes'] == len(dataset)
    ids = list(dataset.class_ids)
    names = getattr(dataset,'class_ids_ori',getattr(dataset,'cat_part_name',ids))
    inter = meter.intersection_buf[1,ids].cpu().tolist()
    union = meter.union_buf[1,ids].cpu().tolist()
    per_class = [{'class_id':int(cid),'original_class_id':str(name),
                  'intersection':i,'union':u,'iou_percent':i/max(u,1)*100}
                 for cid,name,i,u in zip(ids,names,inter,union)]
    assert len(per_class) == len(ids)
    assert abs(sum(x['iou_percent'] for x in per_class)/len(ids)-miou) < .001
    result = {'status':'completed','dataset':args.dataset,'shots':args.shots,
              'variant':args.variant,'fold':args.fold,'episodes':len(dataset),
              'miou_percent':miou,'class_count':len(ids),'per_class_iou':per_class,
              'reference_count_histogram':capture['references'],
              'box_crop':getattr(dataset,'box_crop',False),
              'seconds':time.perf_counter()-started,
              'peak_gpu_memory_mib':torch.cuda.max_memory_allocated()/1024**2}
    save(output/'result.json',result)
    print('EXTENDED_FOLD_PASS',miou,flush=True)


if __name__ == '__main__':
    main()
