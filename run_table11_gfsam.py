"""Time the published Table 11 baselines on fixed COCO pairs."""
import argparse
import ctypes
from datetime import datetime
import os
import statistics
import sys
from time import perf_counter

from evaluate_extended_segmentation import digest, save
from extended_experiment_spec import LOGS, ROOT, arguments
from run_supplement_evaluation import gpu_slot, gpu_snapshot

COMMIT = 'f227d1d856c6cd33b420b18411e28369365bec02'
SOURCE = ROOT.parent/'downloads'/f'GF-SAM-{COMMIT}'
MATCHER_COMMIT = 'd59b6c761f427d52b1ba00132a6b3488f169d039'
MATCHER_SOURCE = ROOT.parent/'downloads'/f'Matcher-{MATCHER_COMMIT}'
PERSAM_COMMIT = 'a7e87245156e0b8efcba10a15a745e699c679d9d'
PERSAM_SOURCE = ROOT.parent/'downloads'/f'Personalize-SAM-{PERSAM_COMMIT}'
SEGGPT_COMMIT = 'cfec84587da2a2cc4e0b9acebe6fba1666e66477'
SEGGPT_SOURCE = ROOT.parent/'downloads'/f'Painter-{SEGGPT_COMMIT}'/'SegGPT/SegGPT_inference'
DETECTRON2_SOURCE = ROOT.parent/'downloads'/'detectron2-247691a1c292654867deb35cabb27f18d75a1363'
SEGGPT = ROOT/'pretrain/seggpt_vit_large.pth'
SAM = ROOT/'pretrain/sam_vit_h_4b8939.pth'
DINO = ROOT/'pretrain/dinov2_vitl14_pretrain.pth'
WEIGHT_SHA256 = {
    DINO.name:'d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428',
    SAM.name:'a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e',
    SEGGPT.name:'59c74d1993ca50ac0ae968d9bbe5db453f50aa5d5c7c9564cce6011c18b16570',
}


class PerSAM:
    """In-memory form of the released persam.py one-shot path."""
    def __init__(self):
        sys.path[:0] = [str(PERSAM_SOURCE),str(ROOT.parent/'envs/pe-addons')]
        from per_segment_anything import SamPredictor, sam_model_registry

        self.predictor = SamPredictor(sam_model_registry['vit_h'](checkpoint=str(SAM)).cuda())

    def clear(self):
        self.predictor.reset_image()

    def set_reference(self, image, mask):
        import numpy as np

        self.reference = image[0,0].mul(255).byte().permute(1,2,0).cpu().numpy()
        self.mask = np.repeat(mask[0,0].byte().cpu().numpy()[:,:,None],3,axis=2)

    def set_target(self, image):
        self.target = image[0].mul(255).byte().permute(1,2,0).cpu().numpy()

    def predict(self):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from persam import point_selection

        predictor = self.predictor
        mask = predictor.set_image(self.reference,self.mask)
        ref_features = predictor.features.squeeze().permute(1,2,0)
        mask = F.interpolate(mask,size=ref_features.shape[:2],mode='bilinear')[0,0]
        embedding = ref_features[mask>0].mean(0)[None]
        prototype = F.normalize(embedding,dim=-1)
        predictor.set_image(self.target)
        target = predictor.features.squeeze()
        height,width = target.shape[-2:]
        similarity = prototype @ F.normalize(target,dim=0).flatten(1)
        similarity = F.interpolate(similarity.reshape(1,1,height,width),scale_factor=4,mode='bilinear')
        similarity = predictor.model.postprocess_masks(similarity,predictor.input_size,predictor.original_size).squeeze()
        positive,pos_label,negative,neg_label = point_selection(similarity,topk=1)
        points = np.concatenate((positive,negative),axis=0)
        labels = np.concatenate((pos_label,neg_label),axis=0)
        guidance = (similarity-similarity.mean())/similarity.std()
        attention = F.interpolate(guidance[None,None],size=(64,64),mode='bilinear').sigmoid()[None].flatten(3)
        _,_,logits,_ = predictor.predict(point_coords=points,point_labels=labels,multimask_output=False,
                                         attn_sim=attention,target_embedding=embedding[None])
        masks,scores,logits,_ = predictor.predict(point_coords=points,point_labels=labels,
                                                 mask_input=logits[:1],multimask_output=True)
        best = int(np.argmax(scores))
        y,x = np.nonzero(masks[best])
        if not x.size:
            return torch.as_tensor(masks[best])
        box = np.array([[x.min(),y.min(),x.max(),y.max()]])
        masks,scores,_,_ = predictor.predict(point_coords=points,point_labels=labels,box=box,
                                              mask_input=logits[best:best+1],multimask_output=True)
        return torch.as_tensor(masks[np.argmax(scores)])


class SegGPT:
    """Use the released 448-pixel per-image engine with binary prompts."""
    def __init__(self):
        sys.path[:0] = [str(SEGGPT_SOURCE),str(DETECTRON2_SOURCE),str(ROOT.parent/'envs/seggpt-addons'),
                        str(ROOT.parent/'envs/sam3-addons'),str(ROOT.parent/'envs/pe-addons')]
        from seggpt_inference import prepare_model

        self.model = prepare_model(str(SEGGPT),seg_type='semantic').cuda()

    def clear(self):
        self.reference = self.target = self.mask = None

    def set_reference(self, image, mask):
        self.reference = image[0,0].permute(1,2,0).cpu().numpy()
        self.mask = mask[0,0].unsqueeze(-1).expand(-1,-1,3).cpu().numpy()

    def set_target(self, image):
        self.target = image[0].permute(1,2,0).cpu().numpy()

    def predict(self):
        import numpy as np
        from seggpt_engine import imagenet_mean, imagenet_std, run_one_image

        inputs = np.concatenate((self.reference,self.target),axis=0)[None]
        prompt = np.concatenate((self.mask,self.mask),axis=0)[None]
        rgb = run_one_image((inputs-imagenet_mean)/imagenet_std,
                            (prompt-imagenet_mean)/imagenet_std,self.model,'cuda')
        return rgb.mean(-1)>128


def evaluate(output, gpu, smoke, method):
    import random
    import numpy as np
    import torch
    from torch.nn.functional import interpolate
    from torchvision import transforms
    import datasets
    import opts

    config = opts.get_args_parser().parse_args(arguments('coco',0,1,output))
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    dataset = datasets.build_dataset('coco',config)
    episodes = [dataset[index] for index in range(1 if smoke else 12)]
    size = {'matcher':518,'seggpt':448}.get(method,1024)
    preprocess = transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor()])
    prepared = []
    for episode in episodes:
        reference = preprocess(episode['ref_imgs'][0].convert('RGB'))[None,None].cuda()
        target = preprocess(episode['tgt_img'].convert('RGB'))[None].cuda()
        mask = interpolate(episode['ref_masks'][0][None,None].float().cuda(),
                           size=(size,size),mode='nearest')
        prepared.append((reference,mask,target))

    required = {'persam':(SAM,), 'seggpt':(SEGGPT,)}.get(method,(DINO,SAM))
    weight_hashes = {path.name:digest(path) for path in required}
    assert all(value==WEIGHT_SHA256[key] for key,value in weight_hashes.items())
    options = argparse.Namespace(device=torch.device('cuda'),dinov2_size='vit_large',
                                 sam_size='vit_h',dinov2_weights=str(DINO),sam_weights=str(SAM))
    if method == 'seggpt':
        model = SegGPT()
        options = argparse.Namespace(checkpoint=str(SEGGPT),seg_type='semantic')
        name,revision,reference_ms = 'SegGPT',SEGGPT_COMMIT,110
    elif method == 'persam':
        model = PerSAM()
        options = argparse.Namespace(sam_type='vit_h',sam_checkpoint=str(SAM))
        name,revision,reference_ms = 'PerSAM',PERSAM_COMMIT,504
    elif method == 'matcher':
        import inspect
        import textwrap

        sys.path[:0] = [str(MATCHER_SOURCE),str(ROOT.parent/'envs/matcher-addons')]
        from matcher import Matcher as matcher_module

        # NumPy 1.24 removed np.int, which was an alias of the built-in int.
        sampler = matcher_module.RobustPromptSampler
        source = textwrap.dedent(inspect.getsource(sampler.get_mask_scores))
        old = 'point.astype(np.int)'
        assert source.count(old)==1
        namespace = {}
        overlay = source.replace(old,'point.astype(int)')
        exec(compile(overlay,'<matcher-numpy-compatibility>','exec'),vars(matcher_module),namespace)
        sampler.get_mask_scores = namespace['get_mask_scores']
        (output/'matcher-numpy-overlay.py').write_text(overlay)
        # Released COCO one-shot command in GETTING_STARTED.md and main_oss defaults.
        vars(options).update(points_per_side=64,pred_iou_thresh=.88,sel_stability_score_thresh=.90,
            stability_score_thresh=.95,iou_filter=.85,box_nms_thresh=.65,output_layer=3,
            dense_multimask_output=0,use_dense_mask=1,multimask_output=1,num_centers=8,use_box=False,
            use_points_or_centers=True,sample_range=(1,6),max_sample_iterations=64,alpha=1.,beta=0.,
            exp=0.,emd_filter=0.,purity_filter=.02,coverage_filter=0.,use_score_filter=True,
            deep_score_norm_filter=.1,deep_score_filter=.33,topk_scores_threshold=0.,num_merging_mask=9)
        model = matcher_module.build_matcher_oss(options)
        name, revision, reference_ms = 'Matcher', MATCHER_COMMIT, 9000
    else:
        sys.path.insert(0,str(SOURCE))
        from matcher.GFSAM import build_model
        model = build_model(options)
        name, revision, reference_ms = 'GF-SAM', COMMIT, 1030

    def predict(pair):
        reference,mask,target = pair
        model.clear()
        model.set_reference(reference,mask.clone())
        model.set_target(target)
        torch.cuda.synchronize()
        started = perf_counter()
        prediction = model.predict()
        torch.cuda.synchronize()
        elapsed = (perf_counter()-started)*1000
        if method == 'gfsam':
            prediction = prediction[0]
        assert prediction.shape[-2:] == (size,size) and torch.isfinite(prediction).all()
        return elapsed

    before = gpu_snapshot(gpu)
    with torch.inference_mode():
        for _ in range(1 if smoke else 3):
            predict(prepared[0])
        times = []
        for index,pair in enumerate(prepared):
            times.append(predict(pair))
            print(f'{name} pair {index+1}/{len(prepared)}: {times[-1]:.1f} ms',flush=True)
    return {'method':name,'gpu_before':before,'gpu_after':gpu_snapshot(gpu),
            'cases':len(times),'times_ms':times,'median_ms':statistics.median(times),
            'paper_rtx4090_ms':reference_ms,'checkpoint_sha256':weight_hashes,'baseline_commit':revision,
            'runner_sha256':digest(__file__),'input_resolution':size,
            'sam_resolution':None if method=='seggpt' else 1024,
            'arguments':{key:str(value) if key=='device' else value for key,value in vars(options).items()},
            'scope':'Shared A10; COCO fold 0, seed 0; synchronized predictor time, excludes input setup.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',required=True,type=int)
    parser.add_argument('--method',choices=['gfsam','matcher','persam','seggpt'],default='gfsam')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
    stage = 'table11-'+args.method+('-smoke' if args.smoke else '')
    status = LOGS/(stage+'-stage-status.json')
    if status.exists():
        raise FileExistsError(status)
    state = {'status':'waiting_for_gpu','gpu':args.gpu,'pid':os.getpid(),'smoke':args.smoke}
    save(status,state)
    try:
        with gpu_slot(args.gpu,False):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
            output = ROOT/'output'/(stage+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            output.mkdir()
            state.update(status='evaluating',output_directory=str(output))
            save(status,state)
            save(output/'result.json',evaluate(output,args.gpu,args.smoke,args.method))
            state.update(status='completed',result_file=str(output/'result.json'))
            save(status,state)
    except Exception as error:
        state.update(status='failed',error=str(error))
        save(status,state)
        raise


if __name__ == '__main__':
    main()
