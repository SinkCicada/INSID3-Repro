"""Table 3 prototype-normalization control and Figure 7 full-fold parameter sweep."""
import argparse
import ctypes
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import evaluate_extended_segmentation as evaluator
from evaluate_extended_segmentation import digest, save
from extended_experiment_spec import LOGS, ROOT
from run_supplement_evaluation import gpu_slot

DATASETS = ('coco','pascal_part')
RANKS = (0,100,200,300,400,500,600,700,800,900)
TAUS = (.40,.45,.55,.60,.65,.70,.75,.80,.90)
NORMALIZED = 'threshold_055_normalized'


def worker(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
    import models.insid3 as module

    rank, tau = 500, .6
    if args.variant == NORMALIZED:
        rank = 0
        source = inspect.getsource(evaluator.predict_threshold)
        old = 'prototype = reference[:, foreground].mean(dim=1)'
        assert source.count(old) == 1
        overlay = source.replace(old, 'prototype = F.normalize(reference[:, foreground].mean(dim=1), p=2, dim=0)')
        namespace = {}
        exec(compile(overlay,'<normalized-prototype-control>','exec'),namespace)
        module.INSID3.predict_mask = namespace['predict_threshold']
        definition = 'Table 3 reconstruction control: normalize the mean reference prototype to unit L2 norm. Retain original unit patch features, 0.55 cutoff, CRF and released evaluation episodes.'
    elif args.variant.startswith('figure7_rank_'):
        rank = int(args.variant.rsplit('_',1)[-1])
        assert rank in RANKS
        definition = 'Figure 7 one-factor SVD rank sweep with released positional-basis input; tau 0.60.'
    else:
        tau = float(args.variant.rsplit('_',1)[-1])
        assert tau in TAUS
        definition = 'Figure 7 one-factor clustering threshold sweep with full aggregation; rank 500.'

    original_arguments = evaluator.arguments
    def control_arguments(*values):
        options = original_arguments(*values)
        options[options.index('--svd-comps')+1] = str(rank)
        options[options.index('--tau')+1] = str(tau)
        return options
    evaluator.arguments = control_arguments
    evaluator.VARIANTS = [*evaluator.VARIANTS,args.variant]
    original_save = evaluator.save
    def record(path,value):
        if path.name == 'configuration.json':
            value.update(variant_definition=definition,control_runner_sha256=digest(__file__))
            if args.variant == NORMALIZED:
                (path.parent/'ablation_overlay.py').write_text(overlay)
        original_save(path,value)
    evaluator.save = record
    sys.argv = [sys.argv[0],'--dataset',args.dataset,'--fold',str(args.fold),
                '--shots','1','--variant',args.variant,'--gpu',str(args.gpu)]
    evaluator.main()


def plot_sweep(groups,output,complete):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    previous = json.loads((ROOT/'results/table1.json').read_text())
    names = {'coco':'COCO-20i','pascal_part':'PASCAL-Part'}
    colors = {'coco':'#157b9e','pascal_part':'#a85014'}
    fig, axes = plt.subplots(1,2,figsize=(8.4,3.5),layout='constrained')
    for axis,kind,default in zip(axes,('rank','tau'),(500,.6)):
        for dataset in DATASETS:
            original = next(r for r in previous if r['dataset']==names[dataset])
            values = {default: original.get('class_macro_miou_percent',original.get('miou_percent'))}
            if kind == 'rank':
                saved = json.loads((LOGS/f'{dataset}-1shot-no_debias-summary.json').read_text())
                values[0] = saved['class_macro_miou_percent']
            prefix = f'figure7_{kind}_'
            for group in groups:
                if group['dataset']==dataset and group['variant'].startswith(prefix):
                    point = float(group['variant'][len(prefix):])
                    values[point] = group['class_macro_miou_percent']
            points = sorted(values.items())
            axis.plot(*zip(*points),marker='o',color=colors[dataset],label=names[dataset])
        axis.set_xlabel('Debiasing rank' if kind=='rank' else 'Clustering threshold')
        axis.set_ylabel('Class-macro mIoU (%)')
        axis.grid(axis='y',alpha=.2)
        axis.legend(frameon=False)
    fig.suptitle('INSID3 parameter sensitivity' + ('' if complete else ' (completed settings only)'))
    for extension in ('png','pdf'):
        fig.savefig(output/f'figure7.{extension}',dpi=250)
    plt.close(fig)


def stage(args):
    mode = 'threshold-control' if args.experiment=='threshold-control' else 'figure7'
    status = LOGS/(mode+'-stage-status.json')
    if status.exists():
        raise FileExistsError(status)
    if mode == 'threshold-control':
        variants = [NORMALIZED]
    else:
        variants = [f'figure7_rank_{s}' for s in (100,300,700,900,200,400,600,800)]
        variants += [f'figure7_tau_{tau:.2f}' for tau in TAUS if tau != .6]
    state = {'status':'waiting_for_gpu','planned_folds':len(variants)*8,'pid':os.getpid(),
             'jobs':[],'completed_groups':[]}
    output = ROOT/'output'/mode
    output.mkdir(exist_ok=True)
    save(status,state)
    try:
        if mode == 'figure7':
            plot_sweep([],output,False)
        with gpu_slot(args.gpu,False) as gpu:
            state.update(status='evaluating',physical_gpu=gpu)
            save(status,state)
            for variant in variants:
                for dataset in DATASETS:
                    results = []
                    for fold in range(4):
                        name = f'{dataset}-{variant}-fold{fold}'
                        log = LOGS/(name+'.log')
                        job = {'name':name,'status':'running','log':str(log)}
                        state['jobs'].append(job)
                        save(status,state)
                        cmd = [sys.executable,'-u',str(Path(__file__).resolve()),'--variant',variant,
                               '--dataset',dataset,'--fold',str(fold),'--gpu',str(gpu)]
                        with log.open('w') as stream:
                            child = subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
                            job['pid'] = child.pid
                            save(status,state)
                            code = child.wait()
                        job['status'] = 'failed' if code else 'completed'
                        if code:
                            raise RuntimeError(f'{name} failed; see {log}')
                        path = next(line[18:] for line in log.read_text(errors='replace').splitlines() if line.startswith('Output directory: '))
                        job['output_directory'] = path
                        results.append(json.loads((Path(path)/'result.json').read_text()))
                        save(status,state)
                    class_ious = [row['iou_percent'] for result in results for row in result['per_class_iou']]
                    summary = {'dataset':dataset,'variant':variant,'folds':4,
                               'class_macro_miou_percent':statistics.mean(class_ious),
                               'fold_macro_miou_percent':statistics.mean(r['miou_percent'] for r in results)}
                    if mode == 'threshold-control':
                        summary['paper_table3_reference_percent'] = {'coco':44.2,'pascal_part':35.4}[dataset]
                    state['completed_groups'].append(summary)
                    save(status,state)
                    if mode == 'figure7':
                        plot_sweep(state['completed_groups'],output,False)
        state['status'] = 'completed'
        save(output/'result.json',state['completed_groups'])
        if mode == 'figure7':
            plot_sweep(state['completed_groups'],output,True)
        save(status,state)
    except Exception as error:
        state.update(status='failed',error=str(error))
        save(status,state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=['threshold-control','figure7'])
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--variant')
    parser.add_argument('--dataset',choices=DATASETS)
    parser.add_argument('--fold',type=int,choices=range(4))
    args = parser.parse_args()
    worker(args) if args.variant else stage(args)
