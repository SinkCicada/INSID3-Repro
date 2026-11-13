"""Reconstruct the alternative backbone comparisons in Tables 4 and 8."""
import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from evaluate_extended_segmentation import digest, save
from extended_experiment_spec import LOGS, ROOT, arguments
from run_spair_stage import argmax_matches
from run_supplement_evaluation import gpu_slot

SOURCE_COMMIT = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
SOURCE = ROOT.parent / 'downloads' / f'dinov2-{SOURCE_COMMIT}'
WEIGHT = ROOT / 'pretrain/dinov2_vitb14_pretrain.pth'
SHA256 = '0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73'
FRANCA_COMMIT = 'd4fd1839dcbe707d0926d90de45be02cbf8d7efd'
FRANCA_SOURCE = ROOT.parent / 'downloads' / f'Franca-{FRANCA_COMMIT}'
FRANCA_WEIGHT = ROOT / 'pretrain/franca_vitb14_In21k.pth'
FRANCA_SHA256 = '2a0888b5bd5be646704f1ecb8409913a0321db97d0cad00df1e924fd38cf6d2a'
PE_COMMIT = '3e352cca660658d4b5c90f42a7808b11469e4c66'
PE_SOURCE = ROOT.parent / 'downloads' / f'perception_models-{PE_COMMIT}'
PE_WEIGHT = ROOT / 'pretrain/PE-Spatial-L14-448.pt'
PE_SHA256 = '47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1'


def encoder():
    import torch

    model = torch.hub.load(str(SOURCE), 'dinov2_vitb14', source='local', pretrained=False)
    model.load_state_dict(torch.load(WEIGHT, map_location='cpu', weights_only=True), strict=True)
    return model.eval()


def franca():
    import torch

    return torch.hub.load(str(FRANCA_SOURCE),'franca_vitb14',source='local',
                          weights='IN21K',local_state_dict=str(FRANCA_WEIGHT)).eval()


def perception_encoder():
    import torch

    sys.path[:0] = [str(PE_SOURCE),str(ROOT.parent/'envs/pe-addons')]
    from core.vision_encoder.pe import VisionTransformer

    class PatchFeatures(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = VisionTransformer.from_config('PE-Spatial-L14-448',
                pretrained=True,checkpoint_path=str(PE_WEIGHT))
            self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
            self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))

        def get_intermediate_layers(self, images, n, reshape):
            assert n == 1 and reshape
            normalized = (images*self.std+self.mean-.5)*2
            tokens = self.encoder.forward_features(normalized,strip_cls_token=True)
            patch = self.encoder.patch_size
            grid = tuple(size//patch for size in images.shape[-2:])
            return [tokens.transpose(1,2).reshape(images.shape[0],-1,*grid)]

    return PatchFeatures().eval()


def run(state, status, table, smoke):
    import torch
    import models
    import models.insid3 as module
    import opts

    if table == 'table8-pe':
        backbone, weight, checksum, commit, build_encoder = (
            'PE-Spatial-L14-448 final dense features', PE_WEIGHT, PE_SHA256,
            PE_COMMIT, perception_encoder)
    elif table == 'table8-franca':
        backbone, weight, checksum, commit, build_encoder = (
            'Franca ViT-B/14 In21K without RASA', FRANCA_WEIGHT, FRANCA_SHA256,
            FRANCA_COMMIT, franca)
    else:
        backbone, weight, checksum, commit, build_encoder = (
            'DINOv2 ViT-B/14 without registers', WEIGHT, SHA256, SOURCE_COMMIT, encoder)
    assert digest(weight) == checksum
    models._build_encoder = lambda model_size='base': build_encoder()
    output = ROOT / 'output' / (status.name.removesuffix('-stage-status.json') + '_' +
                                datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir()
    state.update(status='evaluating', output_directory=str(output))
    save(status, state)
    size, rank = (672, 20) if table == 'table4' else (896, 500)
    scope = ('Matched 48x48 patch grids and SVD20 with the existing DINOv3 Base run.'
             if table == 'table4' else
             'Matched 64x64 patch grids and fixed Table 6 hyperparameters from DINOv3.')
    save(output/'configuration.json', {
        'backbone':backbone, 'image_size':size, 'svd_components':rank,
        'weights_sha256':checksum, 'backbone_source_commit':commit,
        'insid3_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'scope':scope+' Paper checkpoint and tuned settings are unverified.',
        'runner_sha256':digest(__file__), 'gpu':state['gpu'], 'smoke':smoke,
    })

    if table == 'table4':
        import inference_keypoint_matching as official
        module.kernel_softargmax_get_matches_logits = argmax_matches
        check = json.loads((LOGS/'spair-data-check.json').read_text())
        assert check['status'] == 'PASS' and check['test_pairs'] == 12234
        expected = 4 if smoke else 12234
        evaluator = official.PCKEvaluator
        captured = {}

        class Meter(evaluator):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                captured['meter'] = self

        official.PCKEvaluator = Meter
        modes = ['original','debiased']
    else:
        import inference_segmentation as official
        expected = 4 if smoke else 1000
        modes = list(range(1 if smoke else 4))

    if smoke:
        build = official.build_dataset

        def short_dataset(*a, **kw):
            dataset = build(*a, **kw)
            subset = torch.utils.data.Subset(dataset, range(expected))
            if hasattr(dataset, 'class_ids'):
                subset.class_ids = dataset.class_ids
            return subset

        official.build_dataset = short_dataset

    for mode in modes:
        folder = output / str(mode)
        folder.mkdir()
        job = {'name':str(mode),'status':'running','output_directory':str(folder)}
        state['jobs'].append(job)
        save(status, state)
        if table == 'table4':
            settings = opts.get_args_parser().parse_args([
                '--model-size','base','--image-size',str(size),'--svd-comps',str(rank),
                '--seed','0','--num-workers','0','--data-root',str(ROOT/'data'),
                '--output-dir',str(folder),'--device','cuda'])
            settings.debiased = mode == 'debiased'
            settings.split = 'test'
        else:
            options = arguments('coco',mode,1,folder)
            options[options.index('--model-size')+1] = 'base'
            options[options.index('--image-size')+1] = str(size)
            options[options.index('--svd-comps')+1] = str(rank)
            settings = opts.get_args_parser().parse_args(options)
        save(folder/'arguments.json', vars(settings))
        with torch.inference_mode():
            result = official.main(settings)
        if table == 'table4':
            meter = captured['meter']
            assert len(meter.result['pck0.1']['all']) == expected
            job.update(pairs=expected,pck_percent={
                str(alpha):float(score)*100 for alpha,score in zip(meter.alpha,result)})
        else:
            job.update(fold=mode,episodes=expected,miou_percent=float(result))
        job['status'] = 'completed'
        save(folder/'result.json', job)
        save(status, state)

    if table == 'table4':
        before, after = [job['pck_percent'] for job in state['jobs']]
        state['gain_percentage_points'] = {key:after[key]-before[key] for key in before}
        state['paper_reference_percent'] = {'original':[34.4,50.5,66.8],
                                            'debiased':[34.7,50.8,67.1]}
    else:
        state['miou_percent'] = statistics.mean(job['miou_percent'] for job in state['jobs'])
        state['paper_reference_percent'] = {'table8-dinov2':45.1,'table8-franca':39.2,
                                            'table8-pe':48.4}[table]
    state['status'] = 'completed'
    state['result_file'] = str(output/'result.json')
    save(output/'result.json', state)
    save(status, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('table', choices=['table4','table8-dinov2','table8-franca','table8-pe'])
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert ctypes.CDLL(None).prctl(41,1,0,0,0) == 0
    suffix = '-smoke' if args.smoke else ''
    status = LOGS/f'{args.table}{suffix}-stage-status.json'
    if status.exists():
        raise FileExistsError(status)
    state = {'status':'waiting_for_gpu','gpu':args.gpu,'pid':os.getpid(),'jobs':[]}
    save(status,state)
    try:
        with gpu_slot(args.gpu,False):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
            run(state,status,args.table,args.smoke)
    except Exception as error:
        state.update(status='failed',error=str(error))
        save(status,state)
        raise


if __name__ == '__main__':
    main()
