"""Dataset folds and settings for the 5-shot and ablation evaluations."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT.parent / 'setup-logs'
FOLDS = {'isic': 1, 'lung': 1, 'suim': 1, 'coco': 4,
         'isaid': 3, 'pascal_part': 4, 'paco_part': 4, 'lvis': 10}
REPORT_NAMES = {'pascal_part': 'pascal-part', 'paco_part': 'paco'}
PAPER_5SHOT = {'lvis': 47.2, 'coco': 65.1, 'isic': 63.9, 'suim': 61.7,
               'isaid': 56.9, 'lung': 80.1, 'pascal_part': 57.1, 'paco_part': 46.8}
VARIANTS = ['no_debias', 'seed_only_coarse', 'seed_only_fine', 'cross_only']


def arguments(dataset, fold, shots, output, variant='full'):
    return ['--dataset', dataset, '--fold', str(fold), '--shots', str(shots),
            '--model-size', 'large', '--image-size', '1024',
            '--svd-comps', '0' if variant in ('no_debias', 'threshold_055') else '500',
            '--tau', '0.5' if variant == 'seed_only_coarse' else '0.6',
            '--merge-thresh', '0.2', '--crf-mask-refinement', '--crf-size', '640',
            '--seed', '0', '--num-workers', '0', '--device', 'cuda',
            '--data-root', str(ROOT / 'data'), '--output-dir', str(output),
            '--exp-name', f'{dataset}-{shots}shot-{variant}']
