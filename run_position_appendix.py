"""Produce Figure 8 similarity maps and Figure 9 SPair PCA curves."""
import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import pickle
import random

from evaluate_extended_segmentation import digest, save
from extended_experiment_spec import LOGS, ROOT
from run_dinov2_controls import SHA256, encoder
from run_spair_stage import argmax_matches
from run_supplement_evaluation import gpu_slot

DIMS = (32, 64, 100, 200, 400, 600, 800, 1024)


def build_models(dinov2=False, size='base'):
    import torch
    from models import _HUB_NAMES, _WEIGHTS
    from models.insid3 import INSID3

    weight = ROOT/_WEIGHTS[size]
    cache = Path(torch.hub.get_dir())/'facebookresearch_dinov3_main'
    v3 = torch.hub.load(str(cache),_HUB_NAMES[size],source='local',weights=str(weight)).eval()
    resolution, rank = (768,20) if size=='base' else (1024,500)
    models = {'DINOv3':INSID3(v3,image_size=resolution,svd_components=rank).eval()}
    if dinov2:
        models['DINOv2 Base'] = INSID3(encoder(),image_size=672,svd_components=20).eval()
    return models


def pair_features(model, batch):
    import torch
    import torch.nn.functional as F

    model.reset_state()
    model.set_reference(batch['src_img'])
    model.set_target(batch['trg_img'])
    inputs = torch.cat((model._ref_images,model._tgt_image),dim=0).unsqueeze(0)
    features = F.normalize(model._extract_features(inputs),p=2,dim=2)
    return features[0],model._debias_features(features)[0]


def export(fig, output, name):
    for ext in ('png','pdf'):
        fig.savefig(output/f'{name}.{ext}',dpi=240,bbox_inches='tight')


def similarity(output, dataset):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from utils.keypoints import rescale_points

    models = build_models(dinov2=True)
    first = {}
    for index,category in enumerate(dataset.cls):
        first.setdefault(category,index)
    indices = list(first.values())[:4]
    fig, axes = plt.subplots(len(indices),6,figsize=(13,8),layout='constrained')
    records = []
    for row,index in enumerate(indices):
        batch = dataset[index]
        src = batch['src_kps'][0]
        maps = []
        for model in models.values():
            original,debiased = pair_features(model,batch)
            height,width = original.shape[-2:]
            x,y = rescale_points(src,model._orig_ref_size,(height,width)).round().long().tolist()
            x,y = min(max(x,0),width-1),min(max(y,0),height-1)
            for value in (original,debiased):
                maps.append((value[1] * value[0,:,y,x,None,None]).sum(0).float().cpu().numpy())
        vmin,vmax = float(np.min(maps)),float(np.max(maps))
        axes[row,0].imshow(batch['src_img'])
        axes[row,0].scatter(*src.tolist(),marker='+',color='#00d8ff',linewidths=2)
        axes[row,1].imshow(batch['trg_img'])
        axes[row,0].set_ylabel(batch['category'])
        for ax,values in zip(axes[row,2:],maps):
            image = ax.imshow(values,cmap='viridis',vmin=vmin,vmax=vmax,interpolation='nearest')
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(image,ax=list(axes[row,2:]),shrink=.72)
        records.append({'pair_index':index,'pair':dataset.pair_metadata[index],
                        'keypoint_index':0,'shared_similarity_limits':[vmin,vmax]})
    for ax,title in zip(axes[0],('Reference','Target','DINOv3 original','DINOv3 debiased',
                                  'DINOv2 original','DINOv2 debiased')):
        ax.set_title(title)
    fig.suptitle('Feature similarity at four fixed SPair-71k test keypoints')
    export(fig,output,'figure8-similarity')
    plt.close(fig)
    return {'cases':records,'dinov2_weight_sha256':SHA256,'svd_rank':20,
            'scope':'Fixed SPair test pairs, equal 48 by 48 grids. Paper cases and checkpoint unverified.'}


def train_images(count):
    split = ROOT/'data/COCO2014/splits/trn'
    names = set()
    checksums = {}
    for path in sorted(split.glob('fold*.pkl')):
        checksums[path.name] = digest(path)
        for images in pickle.loads(path.read_bytes()).values():
            names.update(Path(item).name.rsplit('_',1)[-1] for item in images
                         if item.startswith('train2014/'))
    images = ROOT/'data/LVIS/coco/train2017'
    choices = sorted(path for path in images.glob('*.jpg') if path.name in names)
    return random.Random(0).sample(choices,count),len(choices),checksums


def fit_bases(model, paths):
    import torch
    import torch.nn.functional as F
    from utils.data import load_image

    samples = [[],[]]
    rng = torch.Generator().manual_seed(0)
    for index,path in enumerate(paths):
        image,_ = load_image(str(path),model._transform,model.device)
        features = F.normalize(model._extract_features(image.unsqueeze(0)),dim=2)
        patches = torch.randperm(features.shape[-2]*features.shape[-1],generator=rng)[:256]
        for bucket,fmap in zip(samples,(features,model._debias_features(features))):
            bucket.append(fmap[0,0].flatten(1)[:,patches].T.float().cpu())
        if (index+1)%32 == 0:
            print(f'PCA training images {index+1}/{len(paths)}',flush=True)
    bases = []
    for bucket in samples:
        features = torch.cat(bucket).to(model.device)
        features -= features.mean(0,keepdim=True)
        covariance = (features.T @ features / (len(features)-1)).double()
        _, vectors = torch.linalg.eigh(covariance)
        bases.append(vectors.flip(1).float().contiguous())
    return bases


def pca(output, dataset, smoke):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from utils.keypoints import rescale_points
    from utils.metrics import PCKEvaluator

    model = build_models(size='large')['DINOv3']
    paths,pool,checksums = train_images(4 if smoke else 256)
    save(output/'pca-training-images.json',{'seed':0,'available_images':pool,
         'patches_per_image':256,'files':[str(p) for p in paths],'training_split_sha256':checksums})
    bases = fit_bases(model,paths)
    torch.save([base.cpu() for base in bases],output/'pca-bases.pt')
    modes = ('PCA-original','Debias + PCA')
    meters = {(mode,dim):PCKEvaluator(pck_by='image') for mode in modes for dim in DIMS}
    limit = 4 if smoke else len(dataset)
    assert smoke or limit == 12234
    for index in range(limit):
        batch = dataset[index]
        source = batch['src_kps'].cuda()
        truth = batch['trg_kps'].cuda().unsqueeze(0)
        count = torch.tensor([batch['n_pts']])
        threshold = torch.tensor([batch['pck_thresh']])
        features = pair_features(model,batch)
        height,width = features[0].shape[-2:]
        coords = rescale_points(source,model._orig_ref_size,(height,width)).round().long()
        x,y = coords[:,0].clamp(0,width-1),coords[:,1].clamp(0,height-1)
        for mode,fmap,basis in zip(modes,features,bases):
            ref = fmap[0][:,y,x].T.float() @ basis
            target = fmap[1].flatten(1).T.float() @ basis
            for dim in DIMS:
                dots = F.normalize(ref[:,:dim],dim=1) @ F.normalize(target[:,:dim],dim=1).T
                coords = argmax_matches(dots.reshape(1,-1,height,width),.04)
                coords = rescale_points(coords,(height,width),model._orig_tgt_size)
                meters[mode,dim].calculate_pck(truth,coords,count,[batch['category']],threshold)
        if (index+1)%100 == 0:
            print(f'PCA test pairs {index+1}/{limit}',flush=True)
    assert all(len(meter.result['pck0.1']['all'])==limit for meter in meters.values())
    rows = [{'dimension':dim,**{mode:float(meters[mode,dim].get_result()[1])*100 for mode in modes}}
            for dim in DIMS]
    fig,ax = plt.subplots(figsize=(6.2,3.7),layout='constrained')
    for mode,marker in zip(modes,('o','s')):
        ax.plot(DIMS,[row[mode] for row in rows],marker=marker,label=mode)
    ax.set(xlabel='Retained PCA dimensions',ylabel='PCK@0.10 (%)')
    ax.grid(alpha=.25)
    ax.legend()
    export(fig,output,'figure9-pca')
    plt.close(fig)
    return {'pairs':limit,'rows':rows,'scope':'DINOv3 Large 1024/SVD500. Centered training PCA; '
            'uncentered projection at test time as in Eqs. 17–18, L2 normalization, hard argmax. '
            'COCO training subset and exact paper dimension settings unverified.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figure',choices=['figure8','figure9'])
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    suffix = '-smoke' if args.smoke else ''
    stage = args.figure+suffix
    path = LOGS/(stage+'-stage-status.json')
    if path.exists():
        raise FileExistsError(path)
    assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
    state = {'status':'waiting_for_gpu','gpu':args.gpu,'smoke':args.smoke,'pid':os.getpid()}
    save(path,state)
    try:
        with gpu_slot(args.gpu,False):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
            import torch
            from datasets.spair import DatasetSPair

            torch.manual_seed(0)
            torch.backends.cuda.matmul.allow_tf32 = True
            check = json.loads((LOGS/'spair-data-check.json').read_text())
            assert check['status']=='PASS' and check['test_pairs']==12234
            dataset = DatasetSPair(str(ROOT/'data'),split='test')
            output = ROOT/'output'/(stage+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            output.mkdir()
            state.update(status='evaluating',output_directory=str(output))
            save(path,state)
            with torch.inference_mode():
                result = similarity(output,dataset) if args.figure=='figure8' else pca(output,dataset,args.smoke)
            save(output/'result.json',dict(result,runner_sha256=digest(__file__)))
            state.update(status='completed',result_file=str(output/'result.json'))
            save(path,state)
    except Exception as error:
        state.update(status='failed',error=str(error))
        save(path,state)
        raise


if __name__ == '__main__':
    main()
