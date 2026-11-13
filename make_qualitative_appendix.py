"""Render Figure 2 cluster maps and Figure 10 counterparts from saved runs."""
import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path

from evaluate_extended_segmentation import digest, save
from extended_experiment_spec import LOGS, ROOT
from run_supplement_evaluation import gpu_slot, load_model

BENCHMARKS = {
    'coco':'COCO-20i', 'lvis':'LVIS', 'isaid':'iSAID', 'suim':'SUIM',
    'isic':'ISIC', 'lung':'Chest X-ray', 'pascal-part':'PASCAL-Part', 'paco':'PACO-Part',
}


def case_for(name):
    suffix = 'case' if name == 'lung' else 'cases'
    case = json.loads((ROOT/'examples'/f'{name}-demo-{suffix}.json').read_text())
    return case['cases'][0] if 'cases' in case else case


def save_plot(fig, output, stem):
    for extension in ('png','pdf'):
        fig.savefig(output/f'{stem}.{extension}',dpi=240,bbox_inches='tight')


def grouping(output):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from utils.clustering import agglomerative_clustering
    from utils.data import load_image

    fig, axes = plt.subplots(4,2,figsize=(7.2,10),layout='constrained')
    model = load_model(1024,output)
    records = []
    for row,name in enumerate(('coco','lvis','isaid','pascal-part')):
        path = ROOT/case_for(name)['target_image']
        with Image.open(path) as image:
            image = image.convert('RGB')
        inputs,_ = load_image(image,model._transform,model.device)
        features = F.normalize(model._extract_features(inputs.unsqueeze(0))[0,0],dim=0)
        labels = agglomerative_clustering(features.flatten(1).T,model.tau)
        labels = labels.reshape(features.shape[-2:]).cpu().numpy()
        colors = np.random.default_rng(0).uniform(.16,.94,size=(int(labels.max())+1,3))
        axes[row,0].imshow(image)
        axes[row,1].imshow(colors[labels],interpolation='nearest',aspect=image.height/image.width)
        axes[row,0].set_title(BENCHMARKS[name])
        axes[row,1].set_title(f'{len(colors)} clusters')
        for ax in axes[row]:
            ax.axis('off')
        records.append({'dataset':name,'image':str(path),'clusters':len(colors)})
    fig.suptitle('DINOv3 patch grouping with the released clustering function')
    save_plot(fig,output,'figure2-clusters')
    plt.close(fig)
    return {'tau':.6,'image_size':1024,'cases':records}


def gallery(output):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    lookup = {}
    for path in sorted((ROOT/'output').glob('demo_*/result.json')):
        result = json.loads(path.read_text())
        if result.get('status')=='completed':
            key = tuple(str(ROOT/result[field]) for field in ('reference_image','reference_mask','target_image'))
            lookup.setdefault(key,path.parent)
    fig,axes = plt.subplots(4,4,figsize=(12,10),layout='constrained')
    records = []

    def overlay(path, mask_path, color):
        with Image.open(path) as image, Image.open(mask_path) as mask:
            rgb = np.asarray(image.convert('RGB')).astype(np.float32)
            mask = np.asarray(mask.convert('L').resize(image.size,resample=Image.Resampling.NEAREST))>0
        rgb[mask] = .55*rgb[mask]+.45*np.array(color)
        return rgb.astype(np.uint8)

    for index,(name,title) in enumerate(BENCHMARKS.items()):
        case = case_for(name)
        reference,mask,target = (str(ROOT/case[key]) for key in
                                 ('reference_image','reference_mask','target_image'))
        run = lookup[reference,mask,target]
        prediction = run/'predicted_mask.png'
        row,col = divmod(index,2)
        axes[row,col*2].imshow(overlay(reference,mask,(0,200,255)))
        axes[row,col*2+1].imshow(overlay(target,prediction,(165,76,239)))
        axes[row,col*2].set_title(title+' reference')
        axes[row,col*2+1].set_title('INSID3 prediction')
        for ax in axes[row,col*2:col*2+2]:
            ax.axis('off')
        records.append({'dataset':name,'case':case,'prediction':str(prediction),
                        'prediction_sha256':digest(prediction)})
    fig.suptitle('One-shot qualitative results from fixed benchmark examples')
    save_plot(fig,output,'figure10-gallery')
    plt.close(fig)
    return {'cases':records,'selection':'First recorded demo case per dataset; saved predictions.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figure',choices=['figure2','figure10'])
    parser.add_argument('--gpu',type=int)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    output = ROOT/'output'/(args.figure+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir()
    if args.figure == 'figure2':
        assert args.gpu is not None and ctypes.CDLL(None).prctl(41,1,0,0,0)==0
        with gpu_slot(args.gpu,False):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
            import torch
            with torch.inference_mode():
                result = grouping(output)
    else:
        result = gallery(output)
    save(output/'result.json',dict(result,runner_sha256=digest(__file__)))
    save(LOGS/(args.figure+'-stage-status.json'),{'status':'completed',
         'output_directory':str(output),'result_file':str(output/'result.json')})
    print('FIGURE_PASS', output, flush=True)


if __name__ == '__main__':
    main()
