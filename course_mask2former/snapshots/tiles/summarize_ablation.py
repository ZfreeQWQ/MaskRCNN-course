"""Report equal-budget short ablations, with no claim of statistically proven gains."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    paths = {'baseline': args.baseline}
    paths.update({name: args.root / name / 'report_train_100.json'
                  for name in ('crop', 'small_loss', 'boundary', 'all')})
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    baseline = reports['baseline']
    shared = ('dataset_manifest_sha256', 'input_checkpoint_sha256', 'final_iteration',
              'implementation_sha256', 'training_images_after_filter')
    hyperparameters = ('seed', 'iters', 'image_size', 'batch_size', 'lr', 'workers', 'data_root')
    expected_flags = {'baseline': (False, False, False), 'crop': (True, False, False),
                      'small_loss': (False, True, False), 'boundary': (False, False, True),
                      'all': (True, True, True)}
    rows = []
    for name, report in reports.items():
        if report['status'] != 'passed' or report['initial_iteration'] != 0:
            raise ValueError(f'{name} is not a passed independent run')
        for key in shared:
            if report[key] != baseline[key]:
                raise ValueError(f'{name} differs in {key}')
        for key in hyperparameters:
            if report['args'][key] != baseline['args'][key]:
                raise ValueError(f'{name} differs in {key}')
        flags = tuple(report['args'][key] for key in
                      ('small_object_crop', 'small_object_loss', 'boundary_loss'))
        if flags != expected_flags[name]:
            raise ValueError(f'{name} switches do not match its label')
        metrics = report['metrics']
        rows.append(dict(name=name, switches=list(flags),
                         mask_AP=round(metrics['coco/segm_mAP'] * 100, 2),
                         mask_APs=round(metrics['coco/segm_mAP_s'] * 100, 2),
                         delta_APs_vs_equal_budget_baseline=round(
                             (metrics['coco/segm_mAP_s'] - baseline['metrics']['coco/segm_mAP_s']) * 100, 2),
                         checkpoint=report['checkpoint'], report=str(paths[name])))
    result = dict(status='passed', shared_checks='same dataset, starting checkpoint, code hashes, '
                  'seed, train/eval image size, batch, learning rate, iterations and effective training images',
                  iterations=100, training_images_after_filter=baseline['training_images_after_filter'],
                  validation_images=250, evaluation_short_edge=512, rows=rows,
                  interpretation='One seed, only 100 iterations and 250 validation images. '
                  'Small AP differences are exploratory; no statistical or final-course improvement is established.',
                  memory_note='Runner logging resets CUDA peak counters. The legacy peak_allocated_mib '
                  'fields in raw reports are final-window observations, NOT complete-run peak memory. '
                  'No memory or FPS improvement is claimed.')
    destination = args.root / 'summary.json'
    if destination.exists() and json.loads(destination.read_text()) != result:
        raise ValueError('Refusing to replace a different summary')
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
