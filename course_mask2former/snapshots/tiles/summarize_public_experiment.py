"""Aggregate verified COCO development reports without presenting them as gains."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--data-root', default='/workspace/data/coco_subset', type=Path)
    args = parser.parse_args()
    reports = {}
    for name, relative in [('pretrained', 'pretrained/report_eval_100.json'),
                           ('train100', 'train100/report_train_100.json'),
                           ('resume120', 'resume120/report_train_120.json')]:
        value = json.loads((args.root / relative).read_text())
        if value['status'] != 'passed':
            raise ValueError(f'{name} was not verified as passed')
        reports[name] = value
    hashes = {report['dataset_manifest_sha256'] for report in reports.values()}
    scales = {report['args']['image_size'] for report in reports.values()}
    if len(hashes) != 1 or len(scales) != 1:
        raise ValueError('Reports used different data manifests or evaluation image scales')
    resume = reports['resume120']
    train = reports['train100']
    if resume['input_checkpoint_sha256'] != train['checkpoint_sha256']:
        raise ValueError('Resumed run did not load the verified train100 checkpoint')
    if (resume['initial_iteration'], resume['final_iteration']) != (100, 120):
        raise ValueError('Unexpected resume iteration range')
    if resume['initial_optimizer_steps'] != [100] or resume['final_optimizer_steps'] != [120]:
        raise ValueError('AdamW step counters did not resume exactly from 100 to 120')
    data = json.loads((args.data_root / 'manifest.json').read_text())
    result = dict(
        status='passed', dataset_statistics=data['statistics'],
        train_val_overlap=data['leakage_checks'],
        dataset_manifest_sha256=next(iter(hashes)),
        evaluation_short_edge=next(iter(scales)),
        runs={name: {'metrics_raw_0_to_1': report['metrics'],
                     'mask_AP_points': report['metrics']['coco/segm_mAP'] * 100,
                     'mask_APs_points': report['metrics']['coco/segm_mAP_s'] * 100,
                     'elapsed_seconds_including_evaluation': report['seconds_including_evaluation'],
                     'peak_allocated_mib': report['peak_allocated_mib']}
              for name, report in reports.items()},
        resume_verification={'from_iteration': 100, 'to_iteration': 120,
                             'adamw_steps': [100, 120], 'source_sha256_matches': True,
                             'finite_saved_model': True, 'classifier_updated': True},
        interpretation='Real RGB data loading, training, COCO mask evaluation, saving and '
                       'stateful resume passed. Short fine-tuning is an engineering test, '
                       'NOT an algorithm improvement or a full COCO/NYUv2/course benchmark.')
    output = args.root / 'summary.json'
    if output.exists() and json.loads(output.read_text()) != result:
        raise ValueError('Refusing to overwrite a different summary')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
