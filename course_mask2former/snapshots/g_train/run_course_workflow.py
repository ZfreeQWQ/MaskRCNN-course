"""Bounded course workflows: frozen source, GPU smoke, then declared experiments."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from launch_nyu_sweep import IMAGE, SOURCE_FILES, atomic_json, docker_command, gpu_snapshot, sha256

OWNED = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def variants(pair):
    if pair == 'baseline':
        return [('rgb', {}), ('gated_depth', dict(multiscale_depth=True, reliability_gate=True))]
    if pair == 'small_supervision':
        shared = dict(multiscale_depth=True, reliability_gate=True)
        return [('gated_small_weight', dict(shared, small_object_loss=True)),
                ('gated_small_points', dict(shared, course_foreground_points=True))]
    if pair == 'scale_crop':
        shared = dict(multiscale_depth=True, reliability_gate=True, course_large_scale_jitter=True)
        return [('gated_large_scale', dict(shared)),
                ('gated_large_scale_crop', dict(shared, small_object_crop=True))]
    if pair == 'budget_probe':
        return [('gated_depth_long', dict(multiscale_depth=True, reliability_gate=True))]
    if pair == 'small_boundary':
        return [('gated_small_boundary_long', dict(multiscale_depth=True,
                    reliability_gate=True, course_small_boundary=True))]
    raise ValueError('Unknown course experiment pair')


def formal_protocol(pair):
    variants(pair)  # Reject unknown protocols before allocating output/resources.
    if pair in ('budget_probe', 'small_boundary'):
        return dict(stage='budget_diagnostic' if pair == 'budget_probe' else 'small_boundary_ablation',
                    iters=30000, val_interval=5000,
                    keep_checkpoints=6, max_job_seconds=43200)
    return dict(stage='baseline' if pair == 'baseline' else 'small_object_ablation',
                iters=10000, val_interval=5000, keep_checkpoints=2, max_job_seconds=28800)


def validate_gpus(pair, gpus, usage):
    count = len(variants(pair))
    if len(gpus) != count or len(set(gpus)) != count:
        raise ValueError(f'{pair} requires exactly {count} distinct GPU(s)')
    if any(g not in usage or usage[g]['memory_used_mib'] > 512 or usage[g]['utilization'] > 10 for g in gpus):
        raise RuntimeError('Selected GPU unavailable or occupied; refusing to collide with other work')


def run_stage(root, plan, stage, data_root, iters, val_interval, smoke):
    pair = plan.get('experiment_pair', 'baseline')
    protocol = formal_protocol(pair)
    validate_gpus(pair, plan['gpus'], gpu_snapshot())
    output = root / stage
    output.mkdir()
    (output/'logs').mkdir()
    (output/'runs').mkdir()
    (output/'source').symlink_to(root/'source', target_is_directory=True)
    jobs=[]
    for name, flags in variants(pair):
        jobs.append(dict(id=name, seed=20261013, output_rel='runs/'+name,
            flags=dict(flags, functional_smoke=smoke)))
    stage_plan=dict(workspace=plan['workspace'],image=plan['image_id'],
        container_output_root='/workspace/'+str(output.relative_to(Path(plan['workspace']))),
        container_name_prefix='ai-course-'+hashlib.sha256(str(output).encode()).hexdigest()[:8],
        runner_script='run_course_experiment.py',recipe=dict(iters=iters,val_interval=val_interval,
            batch_size=2,image_size=1024,workers=2,lr=1e-5,branch_lr_mult=10.,schedule='decay',
            keep_checkpoints=protocol['keep_checkpoints']),jobs=jobs)
    atomic_json(output/'plan.json',stage_plan)
    processes={}
    records={}
    validate_gpus(pair, plan['gpus'], gpu_snapshot())
    for job,gpu in zip(jobs,plan['gpus']):
        name,cmd=docker_command(stage_plan,job,gpu)
        cmd[cmd.index('--data-root')+1]=data_root
        with (output/'logs'/(job['id']+'.log')).open('xb',buffering=0) as log:
            child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        processes[job['id']]=(child,name,time.monotonic())
        OWNED[name]=child
        records[job['id']]=dict(status='running',gpu=gpu,pid=child.pid,container_name=name,started_at_utc=now())
        atomic_json(output/'status.json',dict(status='running',jobs=records,updated_at_utc=now()))
    while processes:
        for identity,(child,name,start) in list(processes.items()):
            if time.monotonic()-start > (1800 if smoke else protocol['max_job_seconds']):
                subprocess.run(['docker','stop','--time','30',name],timeout=45,check=False)
            code=child.poll()
            if code is None:
                continue
            path=output/'runs'/identity/'report.json'
            report=json.loads(path.read_text()) if path.exists() else {}
            passed=(code==0 and report.get('status')=='passed' and report.get('final_iteration')==iters
                    and report.get('final_optimizer_steps')==[iters]
                    and report.get('test_annotations_used') is False and report.get('test_inputs_used') is False)
            records[identity].update(status='passed' if passed else 'failed',exit_code=code,finished_at_utc=now())
            OWNED.pop(name,None)
            del processes[identity]
        stage_status='running' if processes else 'complete' if all(r['status']=='passed' for r in records.values()) else 'completed_with_failures'
        atomic_json(output/'status.json',dict(status=stage_status,jobs=records,updated_at_utc=now()))
        if processes:
            time.sleep(10)
    if any(r['status']!='passed' for r in records.values()):
        raise RuntimeError(stage+' failed; later stages not launched')
    results={name:json.loads((output/'runs'/name/'report.json').read_text()) for name in records}
    atomic_json(output/'summary.json',dict(status='complete',stage=stage,
        functional_smoke=smoke,results={name:dict(metrics=r['metrics'],training_images=r['training_images'],
        validation_images=r['validation_images'],seconds=r['seconds_including_evaluation'],
        course_point_counters=r.get('course_point_counters'),
        small_boundary_counters=r.get('small_boundary_counters')) for name,r in results.items()},
        note='Smoke metrics are not accuracy evidence.' if smoke else
             'Single-seed budget+schedule diagnostic against archived 10k baseline, NOT equal-budget algorithm comparison.'
             if pair == 'budget_probe' else 'Single-seed fixed-budget comparison, not final course score.'))


def worker(root):
    root=Path(root).resolve()
    plan=json.loads((root/'plan.json').read_text())
    state=dict(status='running',phase='sample_smoke',started_at_utc=now(),pid=os_getpid())
    def update(phase):
        state.update(phase=phase,updated_at_utc=now());atomic_json(root/'workflow_status.json',state)
    def intact():
        if not all(sha256(root/'source'/rel)==digest for rel,digest in plan['source_sha256'].items()):
            raise RuntimeError('Frozen course source changed')
    try:
        data=Path(plan['workspace'])/'data/course_blender_20261013'
        if not plan.get('prepared_only'):
            update('sample_smoke');intact()
            run_stage(root,plan,'sample_smoke','/workspace/data/course_blender_20261013/samples',20,20,True)
            update('waiting_for_download')
            while not (data/'archive/download_complete.json').is_file():
                if (root/'STOP').exists():
                    state.update(status='stopped');update('stopped');return
                progress=data/'archive/download_status.json'
                if progress.exists() and json.loads(progress.read_text())['status']=='failed':
                    raise RuntimeError('Course archive download failed; inspect preserved download status')
                time.sleep(20)
            update('preparing_train_val');intact()
            cmd=[sys.executable,str(root/'source/tools/prepare_course_dataset.py'),
                 '--archive',str(data/'archive/blender.zip'),'--output',str(data/'prepared')]
            subprocess.run(cmd,check=True)
        else:
            complete=json.loads((data/'prepared/prepare_complete.json').read_text())
            if complete['manifest_sha256']!=sha256(data/'prepared/manifest.json'):
                raise ValueError('Existing prepared dataset failed manifest validation')
        if (root/'STOP').exists():
            state.update(status='stopped');update('stopped');return
        update('prepared_data_smoke');intact()
        run_stage(root,plan,'prepared_smoke','/workspace/data/course_blender_20261013/prepared',20,20,True)
        if (root/'STOP').exists():
            state.update(status='stopped');update('stopped');return
        protocol = formal_protocol(plan.get('experiment_pair', 'baseline'))
        stage = protocol['stage']
        update(stage);intact()
        run_stage(root,plan,stage,'/workspace/data/course_blender_20261013/prepared',
                  protocol['iters'],protocol['val_interval'],False)
        intact()
        state.update(status='complete',finished_at_utc=now());update('complete')
    except Exception as exc:
        for name,child in OWNED.items():
            if child.poll() is None:
                subprocess.run(['docker','stop','--time','30',name],timeout=45,check=False)
        state.update(status='failed',error=type(exc).__name__+': '+str(exc),finished_at_utc=now())
        update(state['phase'])
        raise


def os_getpid():
    import os
    return os.getpid()


def start(args):
    workspace=Path(args.workspace).resolve()
    root=Path(args.output).resolve()
    if not root.is_relative_to(workspace/'outputs') or root.exists():
        raise ValueError('Require new output directory under workspace/outputs')
    if args.experiment_pair != 'baseline' and not args.prepared_only:
        raise ValueError('Algorithm ablation requires --prepared-only')
    usage=gpu_snapshot()
    validate_gpus(args.experiment_pair, args.gpus, usage)
    if shutil.disk_usage(workspace).free < 100*2**30:
        raise RuntimeError('Require 100 GiB free space for archive, prepared data and experiment reserve')
    checkpoint=workspace/'checkpoints/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth'
    if sha256(checkpoint)!='41b088b62838c986004f94bab2870395b669ec6bd11b6590369b838b0de06879':
        raise ValueError('Official COCO initialization checkpoint changed')
    reference = None
    if args.experiment_pair != 'baseline':
        boundary_probe = args.experiment_pair == 'small_boundary'
        reference_iters = 30000 if boundary_probe else 10000
        ref = workspace/('outputs/course_budget_20261014_v1/budget_diagnostic/runs/gated_depth_long/report.json'
                         if boundary_probe else 'outputs/course_component_20261013_v2/baseline/runs/gated_depth/report.json')
        baseline = json.loads(ref.read_text())
        expected = dict(iters=reference_iters,val_interval=5000,batch_size=2,image_size=1024,
            seed=20261013,lr=1e-5,branch_lr_mult=10.,schedule='decay',depth_max=10.,
            multiscale_depth=True,reliability_gate=True,small_object_loss=False,
            small_object_crop=False,boundary_loss=False,foreground_points=False)
        if (baseline['status'] != 'passed' or baseline['final_optimizer_steps'] != [reference_iters]
                or baseline.get('final_iteration') != reference_iters
                or baseline['test_annotations_used'] or baseline['test_inputs_used']
                or baseline['source_checkpoint_sha256'] != sha256(checkpoint)):
            raise ValueError(f'Reference RGB-D baseline must have completed {reference_iters} optimizer steps')
        if any(baseline['args'].get(k) != v for k,v in expected.items()):
            raise ValueError('Reference baseline recipe is not paired with this ablation')
        if any(baseline['args'].get(k, False) for k in ('rgb_control', 'course_foreground_points', 'course_large_scale_jitter', 'course_small_boundary')):
            raise ValueError('Reference baseline has additional optimization switches enabled')
        if boundary_probe:
            if baseline['checkpoint_sha256'] != 'a833fd0dccee55dc3ccd9a106a4e226d0cc51e4979b5d21b84ea55d80b443790':
                raise ValueError('Expected predeclared accepted 30k control checkpoint')
            if [row['iteration'] for row in baseline['learning_curve']] != list(range(5000,30001,5000)):
                raise ValueError('Incomplete 30k reference validation curve')
        data = workspace/'data/course_blender_20261013/prepared/annotations'
        for split in ('train','val'):
            if sha256(data/f'instances_{split}.json') != baseline[f'{split}_annotations_sha256']:
                raise ValueError('Ablation annotations differ from completed baseline')
        reference = dict(report=str(ref),report_sha256=sha256(ref),
                         checkpoint_sha256=baseline['checkpoint_sha256'],metrics=baseline['metrics'])
    root.mkdir(parents=True)
    source=root/'source';source.mkdir()
    origin=Path(__file__).parent
    files=[origin/name for name in (*SOURCE_FILES,'run_nyu_v2.py','nyu_v2_plan.py','nyu_v4_plan.py',
                                  'run_course_experiment.py','run_course_workflow.py')]
    files+=list((origin/'rgbd').glob('*.py'))+list((origin/'optimizations').glob('*.py'))
    fingerprints={}
    for path in files:
        rel=path.relative_to(origin);target=source/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);fingerprints[str(rel)]=sha256(target)
    for name in ('course_dataset.py','prepare_course_dataset.py'):
        rel='tools/'+name;target=source/rel;target.parent.mkdir(exist_ok=True)
        shutil.copy2(workspace/'tools'/name,target);fingerprints[rel]=sha256(target)
    image_id=subprocess.run(['docker','image','inspect','--format','{{.Id}}',IMAGE],
        capture_output=True,text=True,check=True).stdout.strip()
    protocol = formal_protocol(args.experiment_pair)
    plan=dict(workspace=str(workspace),gpus=args.gpus,source_sha256=fingerprints,image_id=image_id,
        prepared_only=args.prepared_only, experiment_pair=args.experiment_pair,
        reference_baseline=reference,
        created_at_utc=now(),initial_baseline_budget=30000 if args.experiment_pair == 'small_boundary' else 10000,
        validation_interval=5000,
        formal_protocol=protocol,
        image_size=1024,seed=20261013,
        note=('Single gated RGB-D run from original COCO initialization: 30k steps, LR decay at 18k/24k; '
              'budget and schedule BOTH differ from archived 10k baseline. Final 30k endpoint is primary; '
              'six validations/checkpoints, no early stopping or automatic further experiments. '
              if args.experiment_pair == 'budget_probe' else
              'Single small-only boundary treatment against accepted 30k control. Fresh COCO, 30k steps, '
              '18k/24k decay, final30k primary, six full validations, no automatic matrix. '
              if args.experiment_pair == 'small_boundary' else
              'Fixed-budget course pair: '+args.experiment_pair+'. 10k steps at batch2 is about 1.39 train passes. ')
             +'No Kaggle registration, upload or test annotation use.')
    atomic_json(root/'plan.json',plan)
    with (root/'workflow.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u',str(source/'run_course_workflow.py'),'worker','--output',str(root)],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(root/'launch.json',dict(pid=child.pid,started_at_utc=now()))
    print(json.dumps(dict(pid=child.pid,output=str(root)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['start','worker'])
    p.add_argument('--workspace',default='/mnt/nfs/fenghaibin/AI_Class')
    p.add_argument('--output',required=True)
    p.add_argument('--gpus',type=int,nargs='+',default=[0,1])
    p.add_argument('--prepared-only',action='store_true',help='Verify and reuse completed data; start at prepared GPU smoke')
    p.add_argument('--experiment-pair', choices=['baseline', 'small_supervision', 'scale_crop', 'budget_probe', 'small_boundary'], default='baseline')
    args=p.parse_args()
    worker(args.output) if args.action=='worker' else start(args)
