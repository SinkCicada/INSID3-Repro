"""Verify SPair-71k and run both official DINOv3-L correspondence variants.

Uses the released entrypoint defaults (1024 pixels, SVD rank 500), rather
than assuming that the README's Base/768/rank20 example describes Large.
No claim of exact Table 2 configuration equivalence is made.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
LOGS = ROOT.parent / 'setup-logs'
STATUS = LOGS / 'spair-stage-status.json'
SHA = 'd145eebe4d7d02ff0b9e2746dd3d72cfcb872df0dcfd6f3227576eb594408809'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def argmax_matches(logits, softmax_temp, sigma=7):
    """Select the highest-scoring target patch and return its (x, y) index."""
    import torch

    width = logits.shape[-1]
    indices = logits.flatten(-2).argmax(dim=-1)
    return torch.stack((indices % width, indices // width), dim=-1).float()


def prepare(archive):
    assert digest(archive) == SHA, 'Uploaded archive SHA-256 mismatch'
    base = ROOT / 'data/SPair-71k'
    assert not base.exists(), 'Existing data must be inspected before replacing'
    staging = ROOT / 'data' / ('spair-staging-' + str(os.getpid()))
    staging.mkdir()
    with tarfile.open(archive, 'r:gz') as package:
        for member in package:
            relative = Path(member.name)
            assert not relative.is_absolute() and '..' not in relative.parts
            assert relative.parts[0] == 'SPair-71k'
            assert member.isdir() or member.isfile(), member.name
            package.extract(member, staging)
    source = staging / 'SPair-71k'
    from PIL import Image
    images = sorted((source / 'JPEGImages').glob('*/*.jpg'))
    assert len(images) == 1800
    for path in images:
        with Image.open(path) as image:
            image.verify()
    pair_names = (source / 'Layout/large/test.txt').read_text().splitlines()
    pair_names = [name.strip() for name in pair_names if name.strip()]
    assert len(pair_names) == 12234 and len(set(pair_names)) == 12234
    categories = Counter()
    for name in pair_names:
        annotation = json.loads((source / 'PairAnnotation/test' / (name + '.json')).read_text())
        cat = annotation['category']
        categories[cat] += 1
        assert len(annotation['src_kps']) == len(annotation['trg_kps']) > 0
        for filename in (name.split('-')[1]+'.jpg', name.split('-')[2].split(':')[0]+'.jpg'):
            assert (source / 'JPEGImages' / cat / filename).is_file()
        box = annotation['trg_bndbox']
        assert max(box[2]-box[0], box[3]-box[1]) > 0
    assert len(categories) == 18
    source.rename(base)
    staging.rmdir()
    from datasets.spair import DatasetSPair
    dataset = DatasetSPair(str(ROOT / 'data'), split='test')
    assert len(dataset) == 12234
    first = dataset[0]
    assert first['n_pts'] > 0
    save(LOGS / 'spair-data-check.json', {
        'status': 'PASS', 'archive_sha256': SHA, 'images': len(images),
        'test_pairs': len(dataset), 'categories': dict(categories),
        'official_loader': 'DatasetSPair; Layout/large/test.txt',
        'dataset_source_sha256': digest(ROOT / 'datasets/spair.py'),
    })
    canonical = archive.with_name('SPair-71k.tar.gz')
    if archive != canonical and not canonical.exists():
        archive.rename(canonical)


def evaluate(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['MPLBACKEND'] = 'Agg'
    os.chdir(ROOT)
    if args.matching_decoder == 'argmax':
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(41, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'Cannot disable THP for this process')
    import torch
    import models
    import opts
    import inference_keypoint_matching as official
    report = json.loads((LOGS / 'spair-data-check.json').read_text())
    assert report['status'] == 'PASS' and report['test_pairs'] == 12234
    cached = Path(torch.hub.get_dir()) / 'facebookresearch_dinov3_main'
    assert (cached / 'hubconf.py').is_file()
    weights = ROOT / models._WEIGHTS['large']
    weight_sha = digest(weights)
    assert weight_sha == '8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035'
    models._build_encoder = lambda model_size='large': torch.hub.load(
        str(cached), models._HUB_NAMES[model_size], source='local',
        weights=str(ROOT / models._WEIGHTS[model_size]))
    prefix = 'spair_argmax_large_' if args.matching_decoder == 'argmax' else 'spair_large_'
    output = ROOT / 'output' / (prefix + args.evaluate + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir()
    print('Output directory:', output, flush=True)
    if args.matching_decoder == 'argmax':
        import models.insid3 as model_module
        model_module.kernel_softargmax_get_matches_logits = argmax_matches
        (output / 'matching_overlay.py').write_text(inspect.getsource(argmax_matches))
    official_args = opts.get_args_parser().parse_args([
        '--model-size', 'large', '--image-size', '1024', '--svd-comps', '500',
        '--seed', '0', '--num-workers', '0', '--data-root', str(ROOT / 'data'),
        '--output-dir', str(output), '--device', 'cuda'])
    official_args.debiased = args.evaluate == 'debiased'
    official_args.split = 'test'
    capture = {}
    OriginalEvaluator = official.PCKEvaluator

    class RecordedEvaluator(OriginalEvaluator):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            capture['evaluator'] = self

    official.PCKEvaluator = RecordedEvaluator
    save(output / 'configuration.json', {
        'official_arguments': vars(official_args), 'physical_gpu': args.gpu,
        'weights_sha256': weight_sha,
        'repository_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'source_sha256': {p: digest(ROOT / p) for p in [
            'inference_keypoint_matching.py', 'models/insid3.py',
            'datasets/spair.py', 'utils/metrics.py', 'run_spair_stage.py']},
        'data_report_sha256': digest(LOGS / 'spair-data-check.json'),
        'matching_decoder': args.matching_decoder,
        'decoder_scope': 'Released feature extraction, source sampling, target rescaling and PCK; argmax replaces only the decoder when selected',
        'configuration_scope': 'Released CLI defaults, DINOv3-L/1024/SVD500; exact Table 2 configuration equivalence unverified',
        'readme_example_differs': 'README example is Base/768/SVD20 and needs another checkpoint',
    })
    started = time.perf_counter()
    with torch.inference_mode():
        pck = official.main(official_args)
    evaluator = capture['evaluator']
    assert len(evaluator.result['pck0.1']['all']) == 12234
    scores = {str(a): float(v)*100 for a,v in zip(evaluator.alpha, pck)}
    result = {
        'status': 'completed', 'dataset': 'SPair-71k', 'variant': args.evaluate,
        'matching_decoder': args.matching_decoder,
        'test_pairs': 12234, 'pck_percent': scores,
        'aggregation': 'Original PCKEvaluator(pck_by=image, avg_by=all); target bounding-box normalization',
        'configuration': 'DINOv3-L / image-size1024 / SVD500 / seed0',
        'paper_metric_difference': None,
        'comparison_note': 'Configuration equivalence to paper Table 2 unverified; compare these two controlled runs directly.',
        'seconds': time.perf_counter()-started,
    }
    save(output / 'per_class_scores.json', evaluator.avg_result_all())
    save(output / 'result.json', result)
    print(json.dumps(result), flush=True)


def stage(args):
    os.chdir(ROOT)
    status_path = LOGS / 'spair-argmax-stage-status.json' if args.matching_decoder == 'argmax' else STATUS
    if args.matching_decoder == 'argmax' and status_path.exists():
        raise FileExistsError(f'Inspect the existing stage before starting another: {status_path}')
    prefix = 'spair-argmax-' if args.matching_decoder == 'argmax' else 'spair-'
    state = {'status':'preparing_data', 'pid':os.getpid(), 'gpus':[args.gpu],
             'started_at':datetime.now(timezone.utc).isoformat(), 'jobs':[]}
    save(status_path, state)
    try:
        report = LOGS / 'spair-data-check.json'
        if not report.exists():
            prepare(args.archive)
        assert json.loads(report.read_text())['archive_sha256'] == SHA
        state['status'] = 'waiting_for_gpu'; save(status_path, state)
        with (LOGS / f'insid3-gpu{args.gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state['status'] = 'evaluating'; save(status_path, state)
            for variant in ('original', 'debiased'):
                log = LOGS / (prefix + variant + '.log')
                job = {'name':prefix+variant, 'status':'running', 'log':str(log)}
                state['jobs'].append(job); save(status_path, state)
                with log.open('w') as stream:
                    child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                        '--evaluate', variant, '--gpu', str(args.gpu), '--matching-decoder', args.matching_decoder], stdout=stream, stderr=subprocess.STDOUT)
                    job['pid'] = child.pid; save(status_path, state)
                    code = child.wait()
                job['exit_code'] = code
                job['status'] = 'completed' if code == 0 else 'failed'
                for line in log.read_text(errors='replace').splitlines():
                    if line.startswith('Output directory: '):
                        job['output_directory'] = line.split(': ',1)[1]
                if code:
                    save(status_path,state)
                    raise RuntimeError('Failed ' + variant + '; see ' + str(log))
                job['result'] = json.loads((Path(job['output_directory'])/'result.json').read_text())
                save(status_path,state)
        original, debiased = [j['result']['pck_percent'] for j in state['jobs']]
        state.update(status='completed', gains_percentage_points={k:debiased[k]-original[k] for k in original},
                     finished_at=datetime.now(timezone.utc).isoformat())
        if args.matching_decoder == 'argmax':
            soft = json.loads(STATUS.read_text())
            assert soft['status'] == 'completed'
            soft_scores = {j['result']['variant']:j['result']['pck_percent'] for j in soft['jobs']}
            state['argmax_minus_softargmax_percentage_points'] = {
                name:{k:score[k]-soft_scores[name][k] for k in score}
                for name,score in [('original',original),('debiased',debiased)]}
        save(status_path,state)
    except Exception as error:
        state.update(status='failed', error=str(error)); save(status_path,state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--evaluate', choices=['original','debiased'])
    parser.add_argument('--matching-decoder', choices=['softargmax','argmax'], default='softargmax')
    args = parser.parse_args()
    assert args.gpu >= 0
    evaluate(args) if args.evaluate else stage(args)
