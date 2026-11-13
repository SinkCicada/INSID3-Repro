"""Check benchmark data and run 5-shot or ablation evaluations."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from extended_experiment_spec import ROOT, LOGS, FOLDS, REPORT_NAMES, PAPER_5SHOT, VARIANTS, arguments
from evaluate_extended_segmentation import digest, save


def preflight():
    os.chdir(ROOT)
    import random
    import numpy as np
    import torch
    import opts
    from datasets import build_dataset
    records = []
    for key,nfolds in FOLDS.items():
        data_report = LOGS / (REPORT_NAMES.get(key,key)+'-data-check.json')
        if data_report.is_file():
            assert json.loads(data_report.read_text())['status'] == 'PASS', str(data_report)
        for fold in range(nfolds):
            a = opts.get_args_parser().parse_args(arguments(key,fold,5,ROOT/'output'))
            ds = build_dataset(key,args=a)
            cid = list(ds.class_ids)
            assert cid
            required = set()
            pools = []
            if key in ['isic','lung','suim']:
                for pool in ds.img_metadata_classwise.values():
                    pools.append(len(set(map(str,pool))))
                    required.update(map(Path,pool))
                if key == 'isic':
                    for p in list(required):
                        required.add(Path(ds.ann_path)/(p.stem+'_segmentation.png'))
                elif key == 'suim':
                    for p in list(required):required.add(Path(ds.img_path)/(p.stem+'.jpg'))
                else:
                    for p in list(required):
                        name = p.name if 'MCUCXR' in p.name else p.name[:-9]+'.png'
                        required.add(Path(ds.img_path)/name)
            elif key == 'coco':
                for c in cid:
                    pool = ds.img_metadata_classwise[c]
                    pools.append(len(set(pool)))
                    for name in pool:
                        required.add(Path(ds.base_path)/name)
                        required.add(Path(ds.base_path)/'annotations'/Path(name).with_suffix('.png'))
            elif key == 'isaid':
                for c in cid:
                    pool = ds.img_metadata_classwise[c]
                    pools.append(len(set(pool)))
                    for name in pool:
                        required.add(Path(ds.img_path)/(name+'.png'))
                        required.add(Path(ds.ann_path)/(name+'_instance_color_RGB.png'))
            elif key == 'lvis':
                for c in ds.class_ids_ori:
                    pool = ds.img_metadata_classwise[c]
                    pools.append(len(pool))
                    required.update(Path(ds.base_path)/name for name in pool)
            elif key == 'paco_part':
                for c in ds.class_ids_ori:
                    pool = ds.cid2img[c]
                    pools.append(len(pool))
                    for item in pool:
                        name = next(iter(item.values()))
                        required.add(Path(ds.img_path)/'/'.join(name.split('/')[-2:]))
            else:
                for name in ds.cat_part_name:
                    obj,part = name.split('+')
                    pool = ds.cat_annos['object'][obj]['part'][part]['val']
                    pools.append(len(set(pool)))
                    for name in pool:
                        required.add(Path(ds.img_file.format(name)))
                        required.add(Path(ds.anno_file.format(name)))
            missing = sorted(str(p) for p in required if not p.is_file())
            assert not missing, (key,fold,missing[:5])
            assert min(pools) > 1
            random.seed(0);np.random.seed(0);torch.manual_seed(0)
            for index in range(min(2,len(ds))):
                batch = ds[index]
                assert len(batch['ref_imgs']) == len(batch['ref_masks']) == 5
                assert batch['tgt_img'].width > 0 and batch['tgt_img'].height > 0
                assert batch['tgt_mask'].numel() > 0
                assert all(x.numel() > 0 for x in batch['ref_masks'])
            record = {'dataset':key,'fold':fold,'shots':5,'class_count':len(cid),
                      'class_ids':list(map(int,cid)),
                      'original_class_ids':list(map(str,getattr(ds,'class_ids_ori',getattr(ds,'cat_part_name',cid)))),
                      'episodes':len(ds),'min_unique_pool_size':min(pools),
                      'required_existing_files':len(required),
                      'data_report_sha256':digest(data_report) if data_report.is_file() else None,
                      'loader_sha256':digest(ROOT/'datasets'/f'{key}.py'),
                      'sampling':'Unmodified released loader; no forced uniqueness beyond its own rule',
                      'smoke_samples':2}
            records.append(record)
            save(LOGS/'five-shot-preflight-progress.json',{'status':'checking','folds':records})
            print('PREFLIGHT_FOLD_PASS',key,fold,len(cid),len(ds),len(required),flush=True)
    summary = {key:{'folds':sum(r['dataset']==key for r in records),
                    'classes':sum(r['class_count'] for r in records if r['dataset']==key),
                    'episodes':sum(r['episodes'] for r in records if r['dataset']==key)} for key in FOLDS}
    save(LOGS/'five-shot-data-check.json',{
        'status':'PASS','created_at':datetime.now(timezone.utc).isoformat(),
        'datasets':summary,'folds':records,'new_downloads_required':[],
        'paper_scope_note':'LVIS and PACO classes are filtered by shot in the released loader. Counts below describe this code, not an assumed identical paper population.',
        'source_policy':'Original dataset and model files are not modified.',
    })
    print('FIVE_SHOT_DATA_PASS',json.dumps(summary),flush=True)


def run(args):
    os.chdir(ROOT)
    report = json.loads((LOGS/'five-shot-data-check.json').read_text())
    assert report['status'] == 'PASS'
    if args.mode == 'five-shot':
        specifications = [(d,f,5,'full') for d,n in FOLDS.items() for f in range(n)]
    else:
        variants = ['threshold_055'] if args.mode == 'threshold-baseline' else VARIANTS
        specifications = [(d,f,1,v) for v in variants for d in ['coco','pascal_part'] for f in range(FOLDS[d])]
    status_path = LOGS / (args.mode+'-stage-status.json')
    assert not status_path.exists(), 'An existing stage must be inspected before resuming or replacing it'
    state = {'status':'waiting_for_gpu','pid':os.getpid(),'gpus':[args.gpu],
             'planned_folds':len(specifications),'jobs':[],
             'started_at':datetime.now(timezone.utc).isoformat()}
    save(status_path,state)
    try:
        with (LOGS/f'insid3-gpu{args.gpu}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state['status'] = 'evaluating';save(status_path,state)
            for dataset,fold,shots,variant in specifications:
                name = f'{dataset}-{shots}shot-{variant}-fold{fold}'
                log = LOGS/(name+'.log')
                job = {'name':name,'dataset':dataset,'fold':fold,'shots':shots,
                       'variant':variant,'status':'running','log':str(log)}
                state['jobs'].append(job);save(status_path,state)
                command = [sys.executable,'-u',str(ROOT/'evaluate_extended_segmentation.py'),
                           '--dataset',dataset,'--fold',str(fold),'--shots',str(shots),
                           '--variant',variant,'--gpu',str(args.gpu)]
                with log.open('w') as stream:
                    child = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                    job['pid'] = child.pid;save(status_path,state)
                    code = child.wait()
                job['exit_code'] = code;job['status'] = 'completed' if code == 0 else 'failed'
                for line in log.read_text(errors='replace').splitlines():
                    if line.startswith('Output directory: '):job['output_directory'] = line.split(': ',1)[1]
                if code:
                    save(status_path,state)
                    raise RuntimeError(name+' failed; inspect '+str(log))
                result_path = Path(job['output_directory'])/'result.json'
                result = json.loads(result_path.read_text())
                assert result['status'] == 'completed'
                job['result'] = {k:v for k,v in result.items() if k != 'per_class_iou'}
                save(status_path,state)
                group = [j for j in state['jobs'] if j['dataset']==dataset and j['variant']==variant and j['status']=='completed']
                if len(group) == FOLDS[dataset]:
                    results = [json.loads((Path(j['output_directory'])/'result.json').read_text()) for j in group]
                    classes = [c for r in results for c in r['per_class_iou']]
                    class_mean = sum(c['iou_percent'] for c in classes)/len(classes)
                    fold_mean = sum(r['miou_percent'] for r in results)/len(results)
                    summary = {'status':'completed','dataset':dataset,'shots':shots,'variant':variant,
                               'class_macro_miou_percent':class_mean,'fold_macro_miou_percent':fold_mean,
                               'class_count':len(classes),'episodes':sum(r['episodes'] for r in results),
                               'paper_5shot_miou_percent':PAPER_5SHOT[dataset] if shots==5 else None,
                               'paper_difference_percentage_points':None,
                               'comparison_note':'Class population and paper aggregation require review before direct paper comparison.',
                               'fold_outputs':[j['output_directory'] for j in group]}
                    if variant == 'threshold_055':
                        summary['paper_table3_reference_percent'] = {'coco':44.2,'pascal_part':35.4}[dataset]
                        summary['comparison_note'] = 'Eq. (2) reconstruction; original Table 3 normalization, postprocessing and PASCAL group aggregation are unverified.'
                    state.setdefault('completed_groups',[]).append(summary)
                    save(LOGS/f'{dataset}-{shots}shot-{variant}-summary.json',summary)
                    save(status_path,state)
            state.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat())
            save(status_path,state)
    except Exception as error:
        state.update(status='failed',error=str(error));save(status_path,state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['preflight','five-shot','key-ablations','threshold-baseline'])
    parser.add_argument('--gpu',type=int,default=1)
    args = parser.parse_args()
    assert args.gpu >= 0
    preflight() if args.mode == 'preflight' else run(args)
