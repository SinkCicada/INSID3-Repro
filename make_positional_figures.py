"""Produce qualitative Figure 4 and 6 counterparts from fixed real test pairs."""
import argparse
import ctypes
from datetime import datetime
import json
import os

from evaluate_extended_segmentation import digest, save
from extended_experiment_spec import LOGS, ROOT
from run_supplement_evaluation import gpu_slot, load_model


def render(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from torchvision.transforms.functional import normalize
    from datasets.spair import DatasetSPair
    from utils.data import downsample_mask
    from utils.keypoints import rescale_points
    from run_spair_stage import argmax_matches

    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    model = load_model(1024,output)
    records = {}

    def export(fig,name):
        for ext in ('png','pdf'):
            fig.savefig(output/f'{name}.{ext}',dpi=260,bbox_inches='tight')
        plt.close(fig)

    def feature_pair(reference,target,mask=None):
        model.reset_state()
        model.set_reference(reference,mask)
        model.set_target(target)
        inputs = torch.cat((model._ref_images,model._tgt_image),dim=0).unsqueeze(0)
        original = F.normalize(model._extract_features(inputs),p=2,dim=2)
        return original[0],model._debias_features(original)[0]

    def picture(ax,image,title):
        ax.imshow(image)
        ax.set_title(title)
        ax.set_axis_off()

    def heatmaps(fig,axes,target,maps):
        values = np.stack([m.float().cpu().numpy() for m in maps])
        vmin,vmax = float(values.min()),float(values.max())
        width,height = target.size
        for ax,value,title in zip(axes,values,('Original features','Debiased features')):
            picture(ax,target,title)
            im = ax.imshow(value,extent=(-.5,width-.5,height-.5,-.5),cmap='magma',
                           interpolation='bilinear',vmin=vmin,vmax=vmax,alpha=.72)
        fig.colorbar(im,ax=list(axes),shrink=.66,label='Feature dot product')
        return [vmin,vmax]

    case = json.loads((ROOT/'examples/coco-demo-cases.json').read_text())['cases'][0]
    with Image.open(case['reference_image']) as image:
        reference = image.convert('RGB')
    with Image.open(case['target_image']) as image:
        target = image.convert('RGB')
    with Image.open(case['reference_mask']) as image:
        mask = torch.from_numpy(np.array(image)>0)
    original,debiased = feature_pair(reference,target,mask)
    foreground = downsample_mask(model._ref_masks.unsqueeze(1),*original.shape[-2:])
    maps = [torch.einsum('chw,c->hw',features[1],features[0,:,foreground].mean(dim=1))
            for features in (original,debiased)]
    fig,axes = plt.subplots(1,4,figsize=(10.3,3.2),layout='constrained')
    picture(axes[0],reference,'Reference + mask')
    axes[0].contour(mask.numpy(),levels=[.5],colors=['#21d1f3'],linewidths=1.4)
    picture(axes[1],target,'Target')
    records['figure4a'] = {'case':case,'shared_color_range':heatmaps(fig,axes[2:],target,maps)}
    fig.suptitle('Region-based feature similarity on a fixed COCO pair')
    export(fig,'figure4a-region')

    ds = DatasetSPair(str(ROOT/'data'),split='test')
    indices = []
    seen = set()
    for index,category in enumerate(ds.cls):
        if category not in seen:
            seen.add(category)
            indices.append(index)
        if len(indices)==4:
            break
    fig6,axes6 = plt.subplots(len(indices),3,figsize=(8.1,8.2),layout='constrained')
    records['figure6'] = []
    for row,index in enumerate(indices):
        batch = ds[index]
        reference,target = batch['src_img'],batch['trg_img']
        original,debiased = feature_pair(reference,target)
        height,width = original.shape[-2:]
        src = batch['src_kps'][0]
        gt = batch['trg_kps'][0]
        x,y = rescale_points(src,model._orig_ref_size,(height,width)).round().long().tolist()
        x,y = min(max(x,0),width-1),min(max(y,0),height-1)
        maps = [torch.einsum('chw,c->hw',features[1],features[0,:,y,x]) for features in (original,debiased)]
        points = [rescale_points(argmax_matches(m[None,None],.04),(height,width),model._orig_tgt_size)[0,0].cpu() for m in maps]
        errors = [float((point-gt).norm()/batch['pck_thresh']) for point in points]
        picture(axes6[row,0],reference,batch['category']+' reference')
        axes6[row,0].scatter(*src.tolist(),marker='+',s=58,color='#00e5ff',linewidths=2)
        for ax,point,error,title in zip(axes6[row,1:],points,errors,('Original','Debiased')):
            picture(ax,target,f'{title}, error {error:.1%}')
            ax.scatter(*gt.tolist(),marker='o',s=47,facecolors='none',edgecolors='#28dc78',linewidths=1.7)
            ax.scatter(*point.tolist(),marker='x',s=47,color='#ff4236',linewidths=1.7)
        records['figure6'].append({'index':index,'pair':ds.pair_metadata[index],'category':batch['category'],
            'keypoint_index':0,'target_truth_xy':gt.tolist(),'predicted_xy':[p.tolist() for p in points],
            'normalized_errors':errors})
        if row==0:
            fig,axes = plt.subplots(1,4,figsize=(10.3,3.2),layout='constrained')
            picture(axes[0],reference,'Source keypoint')
            axes[0].scatter(*src.tolist(),marker='+',s=60,color='#00e5ff',linewidths=2)
            picture(axes[1],target,'Target')
            records['figure4b'] = {'pair':ds.pair_metadata[index],'shared_color_range':heatmaps(fig,axes[2:],target,maps)}
            fig.suptitle('Keypoint-based feature similarity on a fixed SPair-71k pair')
            export(fig,'figure4b-keypoint')
    fig6.suptitle('SPair-71k matches  |  green circle: truth  |  red cross: prediction')
    export(fig6,'figure6-matches')

    size = model.image_size
    shape = (1,3,size,size)
    gradient = torch.linspace(0,1,size).reshape(1,1,1,size).expand(shape)
    noise = torch.randn(shape,generator=torch.Generator().manual_seed(0))
    inputs = [torch.zeros(shape),gradient,noise]
    projected = []
    for image in inputs:
        normalized = normalize(image,[.485,.456,.406],[.229,.224,.225]).cuda()
        fmap = F.normalize(model._extract_features(normalized.unsqueeze(0))[0,0].float(),p=2,dim=0)
        centered = fmap - fmap.mean(dim=(-2,-1),keepdim=True)
        projected.append(torch.einsum('cr,chw->hwr',model.positional_basis[:,:3],centered).cpu().numpy())
    pc = np.stack(projected)
    low,high = np.quantile(pc,[.01,.99],axis=(0,1,2))
    rgb = np.clip((pc-low)/np.maximum(high-low,1e-8),0,1)
    fig,axes = plt.subplots(2,3,figsize=(7.1,4.8),layout='constrained')
    for col,(name,image,view) in enumerate(zip(('Constant','Gradient','Gaussian noise'),inputs,rgb)):
        display = image[0].permute(1,2,0).numpy()
        picture(axes[0,col],np.clip(display*.18+.5,0,1) if col==2 else display,name)
        picture(axes[1,col],view,'Shared PC 1–3')
    fig.suptitle('Spatial structure in low-semantic inputs')
    export(fig,'figure4c-position-pca')
    records['figure4c'] = {'inputs':['constant zero','horizontal gradient','Gaussian seed 0'],
                           'basis':'Top three directions of the released positional basis, estimated with the constant input',
                           'shared_channel_limits':[low.tolist(),high.tolist()]}
    model.reset_state()
    save(output/'figure_cases.json',records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
    with gpu_slot(args.gpu,False) as gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        import torch
        output = ROOT/'output'/('figures4-6_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        output.mkdir()
        with torch.inference_mode():
            render(output)
        save(output/'configuration.json',{'backbone':'DINOv3-L','image_size':1024,'svd_rank':500,
            'decoder':'maximum similarity patch','case_selection':'First fixed COCO fold 0 demo, then first SPair test pair of each of first four encountered categories, point index zero',
            'implementation_sha256':digest(__file__)})
        print('FIGURES_COMPLETE',output,flush=True)


if __name__ == '__main__':
    main()
