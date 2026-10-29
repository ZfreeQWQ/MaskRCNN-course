"""One authorized point-noise ablation; preparation is inside the fixed budget."""
from types import SimpleNamespace
START=1792817340  # 2026-10-24 12:49:00 Asia/Shanghai
CPU_END=START+2*3600
SMOKE_END=START+8100
FORMAL_END=START+58500
EVAL_END=START+61200
DEADLINE=START+64800
GPU_MAX=54000
NAME='course_m2f_pointnoise_20261024_v1'
JOB='pointnoise_long'
OWNER='f4d0be561fc14e9789cab609579bebec'
ARM='pilot'
PARENT='outputs/course_m2f_final_round_20261023_v1'
REFERENCE=PARENT+'/arms/pilot/evaluate_1536'
REFERENCE_SHA='e72adb85ddb752afd7767c51166d80d8a0d7a49cb9045f4835fa96590d61ee59'
CP_RUN=PARENT+'/arms/pilot/formal/runs/pilot_long'
CP_REPORT='97382aca7658dd38e7f23cad911ac00fddf87e85cddd91f8b22de1f8d997d4c2'
CP_WEIGHT='b1c603d0f0f0866ad23370db1110a12afccdb4ee89ca4bf37b19a281e0600fbd'
EXACT=PARENT+'/arms/pilot/new_exact.json'
EXACT_CACHE_SHA='48de835de0871f9265189f7e1b779274a0d7f2be2f13c44b48ba671ca3b07d38'
AUX_EXACT='outputs/course_m2f_channel_20261023_v2/reference_exact.json'
ACTIVE=44400414
STATE=177814584
ADDED=0
LOCKS={0:'.course_directions_gpu0.lock',1:'.course_maskrcnn_gpu1_formal.lock'}
NEW=('m2f_pointnoise_spec.py','m2f_pointnoise_worker.py','launch_m2f_pointnoise.py',
     'rgbd/course_pointnoise.py','tests/test_m2f_pointnoise.py')
OPTIONS=dict(enabled=True,max_queries=16,weight=.5,no_noise=False,point_noise_ratio=.2)

def job(smoke=False):
    return dict(id=JOB,seed=20261013,output_rel='runs/'+JOB,
        flags=dict(multiscale_depth=True,reliability_gate=True,course_copy_paste=True,functional_smoke=smoke),
        max_job_seconds=50400)

def lease_end(now,used,absolute,limit):
    hard=min(absolute,now+min(limit,GPU_MAX-used))
    if hard-now<=60:raise TimeoutError('Insufficient immutable point-noise GPU budget')
    return hard,hard-20

def arm_spec(root=None,arm=None):
    return SimpleNamespace(**{k:v for k,v in globals().items() if k.isupper()},job=job,lease_end=lease_end)
