"""One approved geometry training; single and system controls fixed separately."""
from types import SimpleNamespace
START=1793068374
CPU_END=1793082774
SMOKE_END=1793083674
FORMAL_END=1793148474
EVAL_END=1793152074
DEADLINE=1793176374
SINGLE_END=1793154774
SYSTEM_END=1793172774
GPU_MAX=86400
NAME='course_m2f_G_20261027_recovery2'
PREVIOUS='outputs/course_m2f_G_20261027_recovery1'
ORIGINAL='outputs/course_m2f_G_20261027_v1'
ROI_PROOF='outputs/G_roi_repair_20261027/report.json'
JOB='geometry_long'
OWNER='7e6d08953f3d4cf8b6a5a24702c967e4'
ARM='pilot'
PARENT='outputs/course_m2f_lr_20261026_v1'
REFERENCE=PARENT+'/evaluate_1536'
REFERENCE_SHA='b920baca26f7c7bf7757eb8e3a3a11e2526f9d1aaddac946ad515adf366d230e'
CP_RUN=PARENT+'/formal/runs/lr3e5_long'
CP_REPORT='22f96a98697ecb7871c8690f40d4c755170e68d6ace3c7899eeeed30f5bac8b3'
CP_WEIGHT='b3a2e8e4420ff397e331d31e1ff58bcaae0ceddfb33794a09872ad27c09ccbb3'
EXACT=PARENT+'/new_exact.json'
EXACT_SHA='31a42dcafde904a8aab00b5972913a94a45eb7f0a3c307acdd27e25add8b20a0'
EXACT_CACHE_SHA='530cd814e3dfe97c6fa2b67a4d38de69cf054b4c7f99634f9913d6cd833579a8'
ACTIVE=0
STATE=0
ADDED=0
LR=3e-5
BASE_LR=1e-5
LOCKS={0:'.course_directions_gpu0.lock'}
UUID_LOCK='.course_gpu7_GPU-3c522036-79d8-7200-09b3-a394915e62df.lock'
UUID='GPU-3c522036-79d8-7200-09b3-a394915e62df'
HOSTNAME='m6-3-a800'
PRIVATE_LOCK='/tmp/sparsekg_sm80_gpu_benchmark.lock'
IMAGE='sha256:eff8b304fe898a9509ee7b6df88b17446519ec29c85d007c6f6ac709787e8140'
OPTIONS=dict(enabled=True,max_queries=16,weight=.5,no_noise=False,point_noise_ratio=.2)
GEO_OPTIONS=dict(enabled=True,dropout=.2,seed=20261013)
NEW=('geometry_spec.py','geometry_signal.py','geometry_worker.py','geometry_launcher.py','geometry_cpu.py','rgbd/course_geometry_model.py','rgbd/course_geometry_data.py','tests/test_geometry.py')
def job(smoke=False):
    return dict(id=JOB,seed=20261013,output_rel='runs/'+JOB,flags=dict(multiscale_depth=True,reliability_gate=True,course_copy_paste=True,functional_smoke=smoke),max_job_seconds=64800)
def lease_end(now,used,absolute,limit):
    hard=min(absolute,DEADLINE,now+min(limit,GPU_MAX-used))
    if hard-now<=60:raise TimeoutError('No remaining geometry budget')
    return hard,hard-20

def arm_spec(root=None,arm=None):return SimpleNamespace(**{k:v for k,v in globals().items() if k.isupper()},job=job,lease_end=lease_end)
