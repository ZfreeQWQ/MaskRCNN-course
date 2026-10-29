"""Arm B includes preparation, has fixed point-noise primary control."""
from types import SimpleNamespace
START=1792909535
CPU_END=1792916735
SMOKE_END=1792917635
FORMAL_END=1792968935
EVAL_END=1792971635
DEADLINE=1792974335
GPU_MAX=54000
ROUND_END=1793002252
ROUND_GPU_MAX=82800
PRIOR_GPU=10287.900446891785
BATCH='course_m2f_abc_20261025_v1'
NAME=BATCH+'/arms/B'
JOB='tiny_matching_long'
OWNER='bdfc28c7bc634e6c929555b2883eb728'
ARM='pilot'
PARENT='outputs/course_m2f_pointnoise_20261024_v1'
REFERENCE=PARENT+'/evaluate_1536'
REFERENCE_SHA='2ab5e154a7ccad7f6ec714b382c9c0803bb4a613c0f3bf28c4174745f754c659'
CP_RUN=PARENT+'/formal/runs/pointnoise_long'
CP_REPORT='0b654113b81565865167e058b320c5f7074fe829a30eb658461078b904e429ce'
CP_WEIGHT='e53ea799348ae53403d15fba07c7d763143f07879191778142a63c2002766ece'
EXACT=PARENT+'/new_exact.json'
EXACT_CACHE_SHA='a8cd22a25e2aa5e8f459cc4bbbb5893fd78d9633d2e57962a4cf191f6daaba0f'
EXACT_SHA='90c94de999e3e8f65947164a0826af25b800d6058d42fe8c78a823ab0f6073eb'
AUX_EXACT='outputs/course_m2f_channel_20261023_v2/reference_exact.json'
ACTIVE=44400414
STATE=177814584
ADDED=0
LOCKS={0:'.course_directions_gpu0.lock'}
NEW=('abc_b_spec.py','abc_b_worker.py','abc_b_launcher.py','rgbd/course_tiny_matching.py','tests/test_abc_b.py')
OPTIONS=dict(enabled=True,max_queries=16,weight=.5,no_noise=False,point_noise_ratio=.2)
def job(smoke=False):
    return dict(id=JOB,seed=20261013,output_rel='runs/'+JOB,flags=dict(multiscale_depth=True,reliability_gate=True,course_copy_paste=True,functional_smoke=smoke),max_job_seconds=50400)
def lease_end(now,used,absolute,limit):
    hard=min(absolute,ROUND_END,now+min(limit,GPU_MAX-used,ROUND_GPU_MAX-PRIOR_GPU-used))
    if hard-now<=60:raise TimeoutError('No remaining B/ABC budget')
    return hard,hard-20
def arm_spec(root=None,arm=None):return SimpleNamespace(**{k:v for k,v in globals().items() if k.isupper()},job=job,lease_end=lease_end)
