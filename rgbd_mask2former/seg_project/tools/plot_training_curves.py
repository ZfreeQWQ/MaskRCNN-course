"""CPU plots from actual Detectron2 JSONL events. No fabricated validation points."""
from __future__ import annotations
import datetime
import json
import math
import re
import statistics
from pathlib import Path

KEYS = ('total_loss', 'lr', 'segm/APs', 'segm/AP', 'time', 'data_time')


def valid(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_events(path):
    """Per-key last recorded event at an iteration wins after a resumed append."""
    series = {key: {} for key in KEYS}
    malformed = 0
    if not Path(path).is_file():
        return series, malformed
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                malformed += 1  # A live process may still be appending its last line.
                continue
            iteration = event.get('iteration') if isinstance(event, dict) else None
            if not valid(iteration) or iteration < 0:
                continue
            completed = int(iteration) + 1
            for key in KEYS:
                value = event.get(key)
                if valid(value):
                    series[key][completed] = float(value)
    return series, malformed


def full_validation_scope(experiment):
    cfg = experiment.get('config', {})
    if cfg.get('PROJECT', {}).get('EVAL_LIMIT') != 0 or cfg.get('DATASETS', {}).get('TEST') != ['blender_val']:
        return False, 'saved config is partial/unknown validation scope'
    # If a run ever used partial eval, JSONL events do not contain timestamps/scope
    # to disambiguate them. Do not silently label those events full-val.
    for path in (Path(experiment['run']) / 'snapshots').glob('*/manifest.json'):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError):
            return False, 'snapshot manifest is incomplete/unreadable; full-val scope unverified'
        limit = manifest.get('arguments', {}).get('eval_limit')
        if limit != 0:
            return False, 'snapshot history contains partial/unknown validation scope'
    return True, 'blender_val, EVAL_LIMIT=0 in saved configuration and snapshot history'


def plot_experiment(experiment, figures, prefix, train_images=14432):
    run = Path(experiment['run'])
    log = run / 'metrics.json'
    series, malformed = read_events(log)
    full_val, val_scope = full_validation_scope(experiment)
    excluded_val_points = sum(len(series[key]) for key in ('segm/APs', 'segm/AP')) if not full_val else 0
    if not full_val:
        series['segm/APs'], series['segm/AP'] = {}, {}
    counts = {key: len(values) for key, values in series.items()}
    time_values = [value for value in series['time'].values() if value > 0]
    batch = experiment.get('batch')
    median_time = statistics.median(time_values) if time_values else None
    result = dict(log=str(log), status='pending', points=counts, malformed_or_partial_lines=malformed,
                  validation_scope=val_scope, excluded_unverified_val_points=excluded_val_points,
                  generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  time_logged_samples=len(time_values), train_step_median_seconds=median_time,
                  global_train_images_per_second=(batch / median_time if valid(batch) and median_time else None),
                  timing_scope='batch / median(recorded time); JSONWriter window-median samples, not all iterations. IterationTimer includes current batch wait+forward/backward/optimizer/DDP; excludes val/checkpoint hooks/process initialization. Not end-to-end throughput or inference benchmark.')
    if not any(counts[key] for key in ('total_loss', 'lr', 'segm/APs', 'segm/AP')):
        result['reason'] = 'training JSONL missing or no finite loss/lr/full-val AP events'
        return result
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as error:
        result.update(status='unavailable', reason=f'matplotlib unavailable: {error}')
        return result
    figures = Path(figures)
    figures.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r'[^A-Za-z0-9_.-]+', '_', run.name)
    output = figures / f'{prefix}_{slug}_training_curves.png'
    epoch_axis = valid(batch) and batch > 0 and valid(train_images) and train_images > 0
    factor = batch / train_images if epoch_axis else 1.0
    xlabel = 'Image-presentations epoch (completed iterations x global batch / train images)' if epoch_axis else 'Completed iterations (iteration + 1)'
    figure = None
    try:
        figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for axis, key, title in zip(axes.flat, ('total_loss', 'lr', 'segm/APs', 'segm/AP'),
                                   ('Training total loss', 'Learning rate', 'Full-val mask APs (%)', 'Full-val mask AP (%)')):
            points = sorted(series[key].items())
            if points:
                x, y = zip(*points)
                # Actual event markers only: no interpolation, zero-fill, moving average or fit.
                axis.scatter([value * factor for value in x], y,
                             s=24 if key.startswith('segm/') else 8, color='#145da0')
            else:
                label = 'No recorded full-val event' if key.startswith('segm/') else 'No finite recorded event'
                axis.text(.5, .5, label, ha='center', va='center', transform=axis.transAxes)
            axis.set_title(f'{title} | {len(points)} recorded points')
            axis.set_xlabel(xlabel, fontsize=8)
            axis.grid(alpha=.2)
            planned = experiment.get('max_iter')
            if valid(planned) and planned > 0:
                axis.set_xlim(0, planned * factor * 1.02)
                axis.axvline(planned * factor, color='gray', linestyle=':', linewidth=.8, label='Planned budget')
            if epoch_axis:
                secondary = axis.secondary_xaxis('top', functions=(lambda value: value/factor, lambda value: value*factor))
                secondary.set_xlabel('Completed iterations', fontsize=8)
        planned_epoch = experiment.get('epochs')
        budget = f'{planned_epoch:.3f}' if valid(planned_epoch) else 'unknown'
        stage = str(experiment.get('stage', 'unknown'))
        if not stage.isascii():
            stage = 'unknown/smoke'
        figure.suptitle(f"{experiment['name']} / {run.name} | stage={stage} | planned={budget} epochs\nActual JSONL events only; full-val panels omit partial/unverified evaluations", fontsize=11)
        figure.savefig(output, dpi=150)
        result.update(status='created', figure=str(output), x_axis='image_presentations_epoch' if epoch_axis else 'completed_iterations')
        sidecar = output.with_suffix('.json')
        result['event_data'] = str(sidecar)
        event_data = dict(result, series={key: [dict(completed_iterations=iteration,
                              image_presentations_epoch=iteration*factor if epoch_axis else None, value=value)
                              for iteration, value in sorted(values.items())] for key, values in series.items()})
        sidecar.write_text(json.dumps(event_data, indent=2, allow_nan=False) + '\n')
    except (OSError, ValueError, RuntimeError) as error:
        result.update(status='unavailable', reason=f'plot generation failed: {type(error).__name__}: {error}')
    finally:
        if figure is not None:
            plt.close(figure)
    return result
