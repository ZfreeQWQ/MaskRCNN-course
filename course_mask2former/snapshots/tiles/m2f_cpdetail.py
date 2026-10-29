"""Exactly one authorized CP+detail arm; reuse frozen production training code."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

JOB_ID = 'copy_paste_detail_long'
FLAGS = dict(multiscale_depth=True, reliability_gate=True,
             course_copy_paste=True, course_detail=True)
REFERENCE = 'outputs/course_m2f_copypaste1536_20261018_v1/evaluate_1536'
REFERENCE_SHA = 'b4e4c3dc6ebfbe3a63e88be36be6065388d63987e92323f7dbde0e57100451ee'
CP_RUN = 'outputs/course_directions_20261015_v1/formal/runs/copy_paste_long'
CP_REPORT_SHA = '790a771d893437dba23e41180efbf149be4cec5398eab733ad924e2d23d14394'


def job(smoke=False):
    return dict(id=JOB_ID, seed=20261013, output_rel='runs/'+JOB_ID,
                flags=dict(FLAGS, functional_smoke=smoke), max_job_seconds=43200)


def check_flags(args):
    if any(getattr(args, k, False) is not v for k,v in FLAGS.items()):
        raise ValueError('Only CP plus detail is authorized')
    if args.course_depth_reliability or args.course_high_resolution:
        raise ValueError('No additional treatment or training scale allowed')


def expected_model(model):
    result = copy.deepcopy(model)
    result.update(type='CourseFeatureRGBDMask2Former', course_detail=dict(enabled=True),
                  course_depth_reliability=dict(enabled=False))
    return result


def train():
    import train_course_directions as training
    from mmengine.config import Config
    from course_demo import sha256
    original_validate = training.validate_direction_args
    original_verify = training.verify_reference_recipe
    def validate(args):
        original_validate(args)
        check_flags(args)
    def verify(args, categories, cfg):
        checks = original_verify(args, categories, cfg)
        cp = Path(args.workspace)/CP_RUN
        if sha256(cp/'report.json') != CP_REPORT_SHA:
            raise ValueError('Accepted CP primary training reference changed')
        reference = Config.fromfile(str(cp/'config.py')).to_dict()
        formal = copy.copy(args)
        formal.functional_smoke = False
        formal.iters, formal.val_interval, formal.keep_checkpoints = 30000, 5000, 6
        actual = training.build_config(formal, categories).to_dict()
        reference['model'] = expected_model(reference['model'])
        reference['optim_wrapper']['paramwise_cfg']['custom_keys']['course_detail_adapter'] = dict(lr_mult=10.,decay_mult=1.)
        keys = ('model','train_cfg','optim_wrapper','param_scheduler','randomness',
                'train_dataloader','val_dataloader','val_evaluator','auto_scale_lr','env_cfg')
        changed = [k for k in keys if actual[k] != reference[k]]
        if changed:
            raise ValueError('Difference beyond approved detail addition to CP: '+str(changed))
        checks.update(cp_reference_report_sha256=CP_REPORT_SHA, cp_plus_detail_only=True)
        return checks
    training.validate_direction_args = validate
    training.verify_reference_recipe = verify
    training.main()


def evaluate(run, output):
    import course_resolution_probe as probe
    import course_directions_diagnostics as diagnostics
    from course_demo import sha256
    from mmengine.config import Config
    diagnostics.jobs = lambda: [job()]
    report, checkpoint, source, size = diagnostics.verify_direction(Path(run))
    if size != 1024 or report['active_parameters'] != 44402718:
        raise ValueError('Unexpected training scale or enabled parameters')
    reference = Path('/workspace')/REFERENCE
    if sha256(reference/'report.json') != REFERENCE_SHA:
        raise ValueError('Primary same-scale CP reference changed')
    old = Config.fromfile(str(reference/'inference_config.py'))
    cfg = Config.fromfile(str(Path(run)/'config.py'))
    if cfg.model.to_dict() != expected_model(old.model.to_dict()):
        raise ValueError('Inference model has differences beyond the approved detail branch')
    if probe.scaled_pipeline(cfg.val_dataloader.dataset.pipeline,1536) != old.val_dataloader.dataset.pipeline:
        raise ValueError('Inference geometry drift')
    def verified(path):
        r,c,s,_ = diagnostics.verify_direction(Path(path))
        return r,c,s
    probe.verify_run = verified
    probe.CHECKPOINT_SHA = report['checkpoint_sha256']
    probe.run(SimpleNamespace(run=run,output=output,mode='evaluate',course_eval_scale=1536))


if __name__ == '__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='evaluate':
        p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--run',required=True);p.add_argument('--output',required=True)
        a=p.parse_args();evaluate(a.run,a.output)
    else:
        train()
