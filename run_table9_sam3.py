"""Evaluate SAM 3 mask propagation and a labeled box-prompt comparison on COCO."""
import argparse
import ctypes
from datetime import datetime
import os
import statistics
import sys

from evaluate_extended_segmentation import digest, save
from extended_experiment_spec import LOGS, ROOT, arguments
from run_supplement_evaluation import gpu_slot

COMMIT = 'f1647e8d0c1d280991f4d1b7d974c2704537fe66'
SOURCE = ROOT.parent/'downloads'/f'sam3-{COMMIT}'
WEIGHT = ROOT/'pretrain/sam3.pt'
WEIGHT_SHA256 = '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'


class SAM3Episode:
    def __init__(self, mode):
        import torch
        from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor

        self.mode = mode
        self.count = 0
        self.torch = torch
        if mode == 'video-mask':
            self.predictor = build_sam3_video_predictor(checkpoint_path=str(WEIGHT),gpus_to_use=[0])
            self.size = self.predictor.model.image_size
        else:
            from sam3.model.sam3_image_processor import Sam3Processor

            model = build_sam3_image_model(checkpoint_path=str(WEIGHT),load_from_HF=False)
            self.predictor = Sam3Processor(model)
            self.size = self.predictor.resolution

    def set_reference(self, image, mask):
        from PIL import Image

        self.reference = image.resize((self.size,self.size),Image.Resampling.BILINEAR)
        self.mask = mask > 0

    def set_target(self, image):
        from PIL import Image

        self.target = image.resize((self.size,self.size),Image.Resampling.BILINEAR)

    def video(self):
        predictor = self.predictor
        request = predictor.handle_request
        session = request({'type':'start_session','resource_path':[self.reference,self.target]})
        session_id = session['session_id']
        try:
            mask = self.torch.nn.functional.interpolate(self.mask[None,None].float(),
                size=(self.size,self.size),mode='nearest')[0,0] > .5
            request({'type':'add_mask','session_id':session_id,'frame_index':0,'obj_id':1,'mask':mask})
            for result in predictor.handle_stream_request({'type':'propagate_in_video',
                'session_id':session_id,'propagation_direction':'forward','start_frame_index':0}):
                if result['frame_index'] == 1:
                    masks = self.torch.as_tensor(result['outputs']['out_binary_masks'],device='cuda')
                    return masks.reshape(-1,self.size,self.size).any(0)
            raise RuntimeError('SAM 3 did not return target frame 1')
        finally:
            request({'type':'close_session','session_id':session_id,'run_gc_collect':False})

    def image_box(self):
        from PIL import Image

        canvas = Image.new('RGB',(2*self.size,self.size))
        canvas.paste(self.reference,(0,0))
        canvas.paste(self.target,(self.size,0))
        points = self.torch.nonzero(self.mask)
        if not points.numel():
            return self.torch.zeros((self.size,self.size),dtype=self.torch.bool,device='cuda')
        top,left = points.min(0).values.tolist()
        bottom,right = (points.max(0).values+1).tolist()
        height,width = self.mask.shape
        box = [(left+right)/(4*width),(top+bottom)/(2*height),
               (right-left)/(2*width),(bottom-top)/height]
        state = self.predictor.set_image(canvas)
        masks = self.predictor.add_geometric_prompt(box,True,state)['masks']
        return masks.reshape(-1,self.size,2*self.size)[:,:,self.size:].any(0)

    def segment(self):
        self.count += 1
        with self.torch.autocast(device_type='cuda',dtype=self.torch.bfloat16):
            return self.video() if self.mode=='video-mask' else self.image_box()


def evaluate(options, state, status):
    import random
    import numpy as np
    import torch
    import inference_segmentation as official
    import opts

    output = ROOT/'output'/(status.name.removesuffix('-stage-status.json')+'_'
                              +datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir()
    state.update(status='evaluating',output_directory=str(output))
    save(status,state)
    assert digest(WEIGHT)==WEIGHT_SHA256
    save(output/'configuration.json',{'source_commit':COMMIT,'weights_sha256':WEIGHT_SHA256,
         'mode':options.mode,'runner_sha256':digest(__file__),'seed':0,
         'prompt_scope':'Reference binary mask on frame 0; forward propagation to frame 1.'
         if options.mode=='video-mask' else
         'Box derived from the reference mask. The released image API accepts boxes, so this is a distinct comparison.'})
    if options.smoke:
        builder = official.build_dataset

        def short_dataset(*a,**kw):
            full = builder(*a,**kw)
            subset = torch.utils.data.Subset(full,range(4))
            subset.class_ids = full.class_ids
            return subset

        official.build_dataset = short_dataset
    expected = 4 if options.smoke else 1000
    with torch.inference_mode():
        model = SAM3Episode(options.mode)
        for fold in range(1 if options.smoke else 4):
            folder = output/f'fold{fold}'
            folder.mkdir()
            args = opts.get_args_parser().parse_args(arguments('coco',fold,1,folder))
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            before = model.count
            miou = official.evaluate(args,model,str(folder/'log.txt'))
            assert model.count-before==expected
            row = {'fold':fold,'episodes':expected,'miou_percent':miou}
            state['folds'].append(row)
            save(folder/'result.json',row)
            save(status,state)
    state.update(status='completed',miou_percent=statistics.mean(f['miou_percent'] for f in state['folds']),
                 result_file=str(output/'result.json'))
    save(output/'result.json',state)
    save(status,state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['video-mask','image-box'],required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    assert WEIGHT.is_file(), f'Missing approved checkpoint at {WEIGHT}'
    assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
    stage = 'table9-'+args.mode+('-smoke' if args.smoke else '')
    status = LOGS/(stage+'-stage-status.json')
    if status.exists():
        raise FileExistsError(status)
    state = {'status':'waiting_for_gpu','gpu':args.gpu,'pid':os.getpid(),'folds':[]}
    save(status,state)
    try:
        with gpu_slot(args.gpu,False):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
            os.environ.setdefault('TRITON_CACHE_DIR',str(ROOT.parent/'cache/triton-sam3'))
            sys.path[:0] = [str(SOURCE),str(ROOT.parent/'envs/sam3-addons'),str(ROOT.parent/'envs/pe-addons')]
            evaluate(args,state,status)
    except Exception as error:
        state.update(status='failed',error=str(error))
        save(status,state)
        raise


if __name__ == '__main__':
    main()
