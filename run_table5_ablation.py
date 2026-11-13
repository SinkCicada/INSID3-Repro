"""Reconstruct Table 5 with the released COCO loader, metrics and INSID3 model."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import evaluate_extended_segmentation as evaluator
from evaluate_extended_segmentation import digest, save
from extended_experiment_spec import LOGS, ROOT
from run_supplement_evaluation import gpu_slot

VARIANTS = {'svd_noise1': 57.6, 'svd_noise5': 57.7, 'svd_noise10': 57.7,
            'augment4': 55.7, 'augment12': 56.5}


def multi_image_basis(model, count):
    import torch
    import torch.nn.functional as F
    from torchvision.transforms.functional import normalize

    def build(device):
        size = model.image_size
        generator = torch.Generator().manual_seed(0)
        shape = (1,3,size,size)
        noise = lambda: torch.randn(shape,generator=generator)
        gradient = torch.linspace(0,1,size).view(1,1,1,size).expand(shape)
        images = [noise()]
        if count >= 5:
            images.extend([torch.zeros(shape),torch.ones(shape),gradient,gradient.transpose(-2,-1)])
        images.extend(noise() for _ in range(count-len(images)))
        features = []
        with torch.no_grad():
            encoder = model.encoder.to(device)
            for image in images:
                image = normalize(image,[.485,.456,.406],[.229,.224,.225]).to(device)
                fmap = encoder.get_intermediate_layers(image,n=1,reshape=True)[0]
                features.append(F.normalize(fmap,p=2,dim=1)[0].flatten(1))
            matrix = torch.cat(features,dim=1)
            matrix -= matrix.mean(dim=1,keepdim=True)
            basis, _, _ = torch.linalg.svd(matrix,full_matrices=False)
        return basis[:,:model.svd_components].contiguous()

    return build


def projective_grid(matrix, height, width):
    import torch

    options = {'device':matrix.device,'dtype':matrix.dtype}
    y = (torch.arange(height,**options)+.5)*2/height-1
    x = (torch.arange(width,**options)+.5)*2/width-1
    yy,xx = torch.meshgrid(y,x,indexing='ij')
    points = torch.stack((xx,yy,torch.ones_like(xx)),dim=-1)
    projected = torch.einsum('bij,hwj->bhwi',matrix,points)
    return projected[...,:2]/projected[...,2:]


def average_view_features(original, views):
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    corners = np.float32([[-1,-1],[1,-1],[1,1],[-1,1]])

    def extract(model, images):
        base = original(model,images)
        batch, frames, channels, height, width = base.shape
        flat_images = images.flatten(0,1)
        total = base.flatten(0,1).float()
        weights = total.new_ones((batch*frames,1,height,width))
        ones = torch.ones_like(weights)
        for _ in range(views-1):
            matrices = []
            for _ in range(batch*frames):
                jitter = rng.uniform(-.08,.08,size=(4,2)).astype(np.float32)
                matrix = cv2.getPerspectiveTransform(corners,corners+jitter)
                flip = np.diag([*rng.choice([-1,1],size=2),1])
                matrices.append(matrix@flip)
            transform = torch.as_tensor(np.stack(matrices),device=images.device,dtype=torch.float32)
            image_grid = projective_grid(torch.linalg.inv(transform),*flat_images.shape[-2:])
            transformed = F.grid_sample(flat_images.float(),image_grid,padding_mode='border',align_corners=False)
            features = original(model,transformed.reshape_as(images)).flatten(0,1).float()
            feature_grid = projective_grid(transform,height,width)
            total += F.grid_sample(features,feature_grid,align_corners=False)
            weights += F.grid_sample(ones,feature_grid,align_corners=False)
        return (total/weights).reshape(batch,frames,channels,height,width).to(base.dtype)

    return extract


def worker(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
    import models.insid3 as module

    variant = args.variant
    if variant.startswith('svd_'):
        count = int(variant.removeprefix('svd_noise'))
        module.INSID3._build_positional_basis = lambda model,device:multi_image_basis(model,count)(device)
        protocol = {'basis_images':count,'basis_recipe':'First seeded N(0,1) image; for 5 or 10 add black, white and two gradients, then additional seeded Gaussian samples as needed; ImageNet normalization; joint centered SVD500'}
    else:
        views = int(variant.removeprefix('augment'))
        module.INSID3._extract_features = average_view_features(module.INSID3._extract_features,views)
        protocol = {'views':views,'view_recipe':'Identity plus seeded independent flips and four-corner perspective jitter in [-0.08,0.08] normalized units; align features back and average using valid support weights; SVD0'}
        original_arguments = evaluator.arguments
        def augmentation_arguments(*values):
            options = original_arguments(*values)
            options[options.index('--svd-comps')+1] = '0'
            return options
        evaluator.arguments = augmentation_arguments
    evaluator.VARIANTS = [*evaluator.VARIANTS,variant]
    original_save = evaluator.save
    def record(path,value):
        if path.name == 'configuration.json':
            value.update(variant_definition=protocol,table5_overlay_sha256=digest(__file__),
                         paper_comparison_note='Reconstructed supplementary protocol. Exact noise or geometric augmentation parameters used for the paper are unverified.')
        original_save(path,value)
    evaluator.save = record
    sys.argv = [sys.argv[0],'--dataset','coco','--fold',str(args.fold),'--shots','1',
                '--variant',variant,'--gpu',str(args.gpu)]
    evaluator.main()


def stage(args):
    status = LOGS/'table5-stage-status.json'
    if status.exists():
        raise FileExistsError(status)
    state = {'status':'waiting_for_table10','planned_folds':20,'pid':os.getpid(),'jobs':[],'completed_groups':[]}
    save(status,state)
    try:
        dependency = LOGS/'table10-stage-status.json'
        while True:
            phase = json.loads(dependency.read_text())['status'] if dependency.exists() else None
            if phase == 'completed':
                break
            if phase == 'failed':
                raise RuntimeError('Table 10 must finish before running Table 5')
            time.sleep(30)
        state['status'] = 'waiting_for_gpu'
        save(status,state)
        with gpu_slot(args.gpu,False) as gpu:
            state.update(status='evaluating',physical_gpu=gpu)
            save(status,state)
            for variant, reference in VARIANTS.items():
                results = []
                for fold in range(4):
                    name = f'table5-{variant}-fold{fold}'
                    log = LOGS/(name+'.log')
                    job = {'name':name,'status':'running','log':str(log)}
                    state['jobs'].append(job)
                    save(status,state)
                    command = [sys.executable,'-u',str(Path(__file__).resolve()),'--variant',variant,
                               '--fold',str(fold),'--gpu',str(gpu)]
                    with log.open('w') as stream:
                        child = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                        job['pid'] = child.pid
                        save(status,state)
                        code = child.wait()
                    job['exit_code'] = code
                    job['status'] = 'failed' if code else 'completed'
                    if code:
                        raise RuntimeError(f'{name} failed; see {log}')
                    prefix = 'Output directory: '
                    output = next(line[len(prefix):] for line in log.read_text(errors='replace').splitlines() if line.startswith(prefix))
                    job['output_directory'] = output
                    result = json.loads((Path(output)/'result.json').read_text())
                    assert result['status']=='completed' and result['episodes']==1000
                    results.append(result['miou_percent'])
                    save(status,state)
                summary = {'variant':variant,'folds':4,'miou_percent':statistics.mean(results),
                           'paper_table5_reference_percent':reference,'scope':'Reconstructed protocol'}
                state['completed_groups'].append(summary)
                save(status,state)
        state['status'] = 'completed'
        save(status,state)
    except Exception as error:
        state.update(status='failed',error=str(error))
        save(status,state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--variant',choices=VARIANTS)
    parser.add_argument('--fold',type=int,choices=range(4))
    args = parser.parse_args()
    worker(args) if args.variant else stage(args)
