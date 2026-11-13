"""Evaluate reconstructed COCO negative pairs or profile INSID3 on an idle A10."""
import argparse
from collections import Counter
from contextlib import contextmanager
import ctypes
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
import zipfile

from evaluate_extended_segmentation import digest, save
from extended_experiment_spec import LOGS, ROOT, arguments

COCO = ROOT / 'data/COCO2014'
ARCHIVE = ROOT.parent / 'downloads/coco-annotations-trainval2014.zip'
ARCHIVE_MD5 = '0a379cfc70b0e71301e0f377548639bd'
PAIR_MANIFEST = LOGS / 'table10-negative-pairs.json'
SIZES = [512, 720, 896, 1024, 1440, 1600, 1760]


def prepare_pairs():
    import numpy as np
    from PIL import Image
    import pickle

    if PAIR_MANIFEST.exists():
        manifest = json.loads(PAIR_MANIFEST.read_text())
        assert manifest['annotation_md5'] == ARCHIVE_MD5 and len(manifest['pairs']) == 4000
        return manifest['pairs']
    final = ARCHIVE.with_suffix('')
    if not final.exists():
        deadline = time.monotonic() + 3600
        while not ARCHIVE.exists() or ARCHIVE.stat().st_size < 252872794:
            if time.monotonic() >= deadline:
                raise TimeoutError('COCO annotation download did not complete')
            time.sleep(15)
        h = hashlib.md5()
        with ARCHIVE.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):
                h.update(block)
        assert h.hexdigest() == ARCHIVE_MD5
        ARCHIVE.rename(final)
    with zipfile.ZipFile(final) as z, z.open('annotations/instances_val2014.json') as stream:
        instances = json.load(stream)
    categories = sorted((entry['id'],entry['name']) for entry in instances['categories'])
    assert len(categories) == 80 and len(instances['images']) == 40504
    images = {entry['id']:entry['file_name'] for entry in instances['images']}
    categories_by_image = {image_id:set() for image_id in images}
    for entry in instances['annotations']:
        categories_by_image[entry['image_id']].add(entry['category_id'])
    image_ids = {name:image_id for image_id,name in images.items()}
    assert all((COCO/'val2014'/name).is_file() for name in images.values())
    pool = sorted(i for i, labels in categories_by_image.items() if labels)
    rng = random.Random(0)
    pairs = []
    for fold in range(4):
        with (COCO/'splits/val'/f'fold{fold}.pkl').open('rb') as stream:
            classes = pickle.load(stream)
        for index in range(fold,80,4):
            category_id, category_name = categories[index]
            refs = sorted({Path(name).name for name in classes[index]
                           if category_id in categories_by_image[image_ids[Path(name).name]]})
            eligible_targets = [i for i in pool if category_id not in categories_by_image[i]]
            for target_id in rng.sample(eligible_targets,50):
                while refs:
                    reference = rng.choice(refs)
                    with Image.open(COCO/'annotations/val2014'/Path(reference).with_suffix('.png')) as label:
                        if np.any(np.asarray(label) == index+1):
                            break
                    refs.remove(reference)
                assert refs and reference != images[target_id]
                pairs.append({'class_index':index,'coco_category_id':category_id,'category':category_name,
                              'reference':reference,'target':images[target_id],
                              'target_annotated_category_ids':sorted(categories_by_image[target_id])})
    assert len(pairs) == 4000 and set(Counter(p['class_index'] for p in pairs).values()) == {50}
    save(PAIR_MANIFEST, {'seed':0,'pairs':pairs,'annotation_md5':ARCHIVE_MD5,
                        'annotation_sha256':digest(final),
                        'protocol':'Reconstructed; 50 distinct annotated negative targets per category. Absence checked against all official instance labels including crowd. Reference drawn from released fold class pool with positive mask.'})
    return pairs


def load_model(size, output):
    import numpy as np
    import torch
    import models
    import opts

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cache = Path(torch.hub.get_dir()) / 'facebookresearch_dinov3_main'
    models._build_encoder = lambda model_size='large':torch.hub.load(
        str(cache), models._HUB_NAMES[model_size], source='local', weights=str(ROOT/models._WEIGHTS[model_size]))
    model_args = opts.get_args_parser().parse_args(arguments('coco',0,1,output))
    model_args.image_size = size
    return models.build_insid3_from_args(model_args).to('cuda').eval()


def evaluate_empty(pairs, output):
    import numpy as np
    import torch
    from PIL import Image

    records = []
    with torch.inference_mode():
        model = load_model(1024,output)
        for index,pair in enumerate(pairs):
            with Image.open(COCO/'annotations/val2014'/Path(pair['reference']).with_suffix('.png')) as image:
                mask = torch.from_numpy(np.asarray(image)==pair['class_index']+1)
            assert mask.any() and pair['coco_category_id'] not in pair['target_annotated_category_ids']
            model.set_reference(str(COCO/'val2014'/pair['reference']),mask)
            model.set_target(str(COCO/'val2014'/pair['target']))
            pixels = int(model.segment().count_nonzero())
            records.append({'pair_index':index,'foreground_pixels':pixels,'correct_empty':pixels==0})
            if (index+1)%50 == 0:
                print(f'negative pairs {index+1}/{len(pairs)}',flush=True)
    counts = Counter(p['category'] for p,r in zip(pairs,records) if r['correct_empty'])
    return {'pairs':len(pairs),'correct_empty':sum(counts.values()),
            'correct_empty_percent':sum(counts.values())/len(pairs)*100,
            'per_category_correct':{p['category']:counts[p['category']] for p in pairs},
            'paper_table10_reference_percent':85,'pair_manifest_sha256':digest(PAIR_MANIFEST),
            'comparison_scope':'Reconstructed negative pairs; paper pair IDs unverified','predictions':records}


def gpu_snapshot(gpu):
    result = subprocess.check_output(['nvidia-smi','-i',str(gpu),
        '--query-gpu=name,utilization.gpu,memory.free,driver_version','--format=csv,noheader,nounits'],text=True)
    name, utilization, free_mib, driver = (item.strip() for item in result.split(','))
    return {'name':name,'utilization_percent':int(utilization),'free_mib':int(free_mib),'driver':driver}


@contextmanager
def gpu_slot(gpu, idle_only):
    while True:
        candidates = [gpu] if gpu is not None else range(4)
        for index in candidates:
            with (LOGS/f'insid3-gpu{index}.lock').open('a') as lock:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if idle_only:
                    samples = []
                    for _ in range(3):
                        samples.append(gpu_snapshot(index))
                        time.sleep(2)
                    if any(s['utilization_percent']>3 or s['free_mib']<21000 for s in samples):
                        continue
                yield index
                return
        time.sleep(30)


def profile(output, gpu, smoke):
    import numpy as np
    import torch
    import datasets
    import models.insid3 as module
    import opts
    from unittest.mock import patch

    args = opts.get_args_parser().parse_args(arguments('coco',0,1,output))
    np.random.seed(0)
    data = datasets.build_dataset('coco',args)
    batches = [data[i] for i in range(1 if smoke else 12)]
    rows = []
    def elapsed(function, *values):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = function(*values)
        torch.cuda.synchronize()
        return result, (time.perf_counter()-started)*1000

    with torch.inference_mode():
        for size in ([1024] if smoke else SIZES):
            before = gpu_snapshot(gpu)
            model = load_model(size,output)
            timings = []
            modules = []
            def prepare(batch):
                model.set_reference(batch['ref_imgs'][0],batch['ref_masks'][0])
                model.set_target(batch['tgt_img'])

            for _ in range(1 if smoke else 3):
                prepare(batches[0]); model.segment()
            torch.cuda.reset_peak_memory_stats()
            for batch in batches:
                prepare(batch)
                _, ms = elapsed(model.segment)
                timings.append(ms)
            if size == 1024:
                observed = {}
                def timed(function,name):
                    def wrapped(*values,**kwargs):
                        result, ms = elapsed(lambda:function(*values,**kwargs))
                        observed[name] = observed.get(name,0)+ms
                        return result
                    return wrapped
                with (patch.object(model,'_extract_features',timed(model._extract_features,'encoder_and_debias')),
                     patch.object(model,'_debias_features',timed(model._debias_features,'encoder_and_debias')),
                     patch.object(module,'agglomerative_clustering',timed(module.agglomerative_clustering,'clustering')),
                     patch.object(model,'_finalize_mask',timed(model._finalize_mask,'crf_and_resize'))):
                    for batch in batches:
                        observed.clear()
                        prepare(batch)
                        _, ms = elapsed(model.segment)
                        observed['similarity_scoring_and_remaining'] = ms-sum(observed.values())
                        observed['profiled_total'] = ms
                        modules.append(dict(observed))
            row = {'size':size,'median_ms':statistics.median(timings),'times_ms':timings,
                   'modules_ms':modules,'peak_gpu_memory_mib':torch.cuda.max_memory_allocated()/1024**2,
                   'gpu_before':before,'gpu_after':gpu_snapshot(gpu)}
            rows.append(row)
            print('PROFILE_SIZE_PASS',size,round(row['median_ms'],2),flush=True)
            del model
            torch.cuda.empty_cache()
            save(output/'resolution_progress.json',rows)
    return {'rows':rows,'protocol':'A10 host measurement; fixed COCO pairs, synchronized wall time, 3 warmups per size; excludes model initialization, positional SVD, image loading and input preparation. Instrumented breakdown measured separately.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',choices=['table10','table12-13'])
    parser.add_argument('--gpu',type=int)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--shared-gpu',action='store_true',
                        help='Record a separate timing run under the current shared GPU load')
    args = parser.parse_args()
    if args.shared_gpu and (args.experiment != 'table12-13' or args.smoke or args.gpu is None):
        parser.error('--shared-gpu requires table12-13, an explicit --gpu, and a full run')
    os.chdir(ROOT)
    assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
    suffix = '-smoke' if args.smoke else '-shared' if args.shared_gpu else ''
    status = LOGS/(args.experiment+suffix+'-stage-status.json')
    if status.exists():
        raise FileExistsError(status)
    state = {'status':'preparing','pid':os.getpid(),'experiment':args.experiment,'smoke':args.smoke,
             'shared_gpu':args.shared_gpu}
    save(status,state)
    try:
        pairs = prepare_pairs() if args.experiment == 'table10' else None
        idle_only = args.experiment == 'table12-13' and not (args.smoke or args.shared_gpu)
        state['status'] = 'waiting_for_idle_gpu' if idle_only else 'waiting_for_gpu'
        save(status,state)
        with gpu_slot(args.gpu,idle_only) as gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
            output = ROOT/'output'/(args.experiment+suffix+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            output.mkdir()
            state.update(status='evaluating',physical_gpu=gpu,output_directory=str(output))
            save(status,state)
            import models
            save(output/'configuration.json',{'model_size':'large','svd_components':500,'seed':0,'crf_size':640,
                'weight_sha256':digest(ROOT/models._WEIGHTS['large']),'runner_sha256':digest(__file__),
                'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'gpu':gpu_snapshot(gpu),'cpu_threads':os.environ.get('OMP_NUM_THREADS')})
            result = (evaluate_empty(pairs[:2] if args.smoke else pairs,output) if pairs is not None
                      else profile(output,gpu,args.smoke))
            result.update(status='completed',smoke=args.smoke,shared_gpu=args.shared_gpu)
            if args.experiment == 'table12-13':
                result['table11_insid3_1024_median_ms'] = next(
                    row['median_ms'] for row in result['rows'] if row['size']==1024)
            if args.shared_gpu:
                result['protocol'] += ' Shared GPU timing, with other server workloads present.'
            save(output/'result.json',result)
            state.update(status='completed',result_file=str(output/'result.json'))
            save(status,state)
    except Exception as error:
        state.update(status='failed',error=str(error))
        save(status,state)
        raise


if __name__ == '__main__':
    main()
