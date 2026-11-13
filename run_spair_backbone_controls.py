"""Run Small, Base and Large with the same released Base settings and argmax."""
import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess

from evaluate_extended_segmentation import digest, save
from extended_experiment_spec import LOGS, ROOT
from run_spair_stage import argmax_matches
from run_supplement_evaluation import gpu_slot

SIZES = ('small','base','large')


def run(args):
    import torch
    import inference_keypoint_matching as official
    import models
    import models.insid3 as module
    import opts

    report = LOGS/'spair-data-check.json'
    data = json.loads(report.read_text())
    assert data['status']=='PASS' and data['test_pairs']==12234
    cached = Path(torch.hub.get_dir())/'facebookresearch_dinov3_main'
    models._build_encoder = lambda model_size='large':torch.hub.load(
        str(cached), models._HUB_NAMES[model_size], source='local',
        weights=str(ROOT/models._WEIGHTS[model_size]))
    module.kernel_softargmax_get_matches_logits = argmax_matches
    if args.smoke:
        build = official.build_dataset
        official.build_dataset = lambda *a,**kw:torch.utils.data.Subset(build(*a,**kw),range(4))
    expected_pairs = 4 if args.smoke else 12234
    sizes = ('large',) if args.smoke else SIZES
    hashes = {}
    for size in sizes:
        weight = ROOT/models._WEIGHTS[size]
        hashes[size] = digest(weight)
        assert hashes[size].startswith(weight.stem.rsplit('-',1)[-1]),weight
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = ROOT/'output'/(('spair-backbones-smoke_' if args.smoke else 'spair-backbones_')+stamp)
    output.mkdir()
    status_path = LOGS/('spair-backbones-smoke-stage-status.json' if args.smoke else 'spair-backbones-stage-status.json')
    state = {'status':'evaluating','gpu':args.gpu,'output_directory':str(output),'jobs':[],'smoke':args.smoke}
    save(status_path,state)
    captured = {}
    original_evaluator = official.PCKEvaluator
    class RecordedEvaluator(original_evaluator):
        def __init__(self,*values,**kwargs):
            super().__init__(*values,**kwargs)
            captured['meter'] = self
    official.PCKEvaluator = RecordedEvaluator
    try:
        for size in sizes:
            scores = {}
            for mode in ('original','debiased'):
                folder = output/f'{size}-{mode}'
                folder.mkdir()
                settings = opts.get_args_parser().parse_args([
                    '--model-size',size,'--image-size','768','--svd-comps','20',
                    '--seed','0','--num-workers','0','--data-root',str(ROOT/'data'),
                    '--output-dir',str(folder),'--device','cuda'])
                settings.debiased = mode=='debiased'
                settings.split = 'test'
                save(folder/'configuration.json',{'arguments':vars(settings),'weights_sha256':hashes[size],
                    'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    'decoder':'hard argmax','scope':'Shared README Base settings (768 / SVD20) across sizes. Paper per-backbone settings unverified.',
                    'data_report_sha256':digest(report),'runner_sha256':digest(__file__)})
                job = {'size':size,'mode':mode,'status':'running','output_directory':str(folder)}
                state['jobs'].append(job)
                save(status_path,state)
                with torch.inference_mode():
                    pck = official.main(settings)
                meter = captured['meter']
                assert len(meter.result['pck0.1']['all'])==expected_pairs
                scores[mode] = {str(alpha):float(value)*100 for alpha,value in zip(meter.alpha,pck)}
                job.update(status='completed',test_pairs=expected_pairs,pck_percent=scores[mode])
                save(folder/'result.json',job)
                save(status_path,state)
            state.setdefault('gains_percentage_points',{})[size] = {
                key:scores['debiased'][key]-scores['original'][key] for key in scores['original']}
            save(status_path,state)
        state['status'] = 'completed'
        save(status_path,state)
        save(output/'result.json',state)
    except Exception as error:
        state.update(status='failed',error=str(error))
        save(status_path,state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
    with gpu_slot(args.gpu,False) as gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        run(args)
