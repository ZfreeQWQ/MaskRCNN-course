"""Authorized single LR ablation; all preparation counts toward 18 hours."""
from types import SimpleNamespace
START=1792990326
CPU_END=START+7200
SMOKE_END=START+8100
FORMAL_END=START+58500
EVAL_END=START+61200
DEADLINE=START+64800
GPU_MAX=54000
NAME='course_m2f_lr_20261026_v1'
JOB='lr3e5_long'
OWNER='347396a77e3b48d0b1fae008e98f1a35'
ARM='pilot'
PARENT='outputs/course_m2f_abc_20261025_v1/arms/B'
REFERENCE=PARENT+'/evaluate_1536'
REFERENCE_SHA='963b806d90d7bb01a52dee7d24607be3b8468c1b2c6fb7391c1cddf43b244ed0'
CP_RUN=PARENT+'/formal/runs/tiny_matching_long'
CP_REPORT='e3c3931c7f983a9c32bae0412ffcdedfdffe7ab0eab8f03444496b83d2bc7eeb'
CP_WEIGHT='33e7c1667184d7325c1c404dc17ec3e7c3041d568976e571af3a1ee7153301e4'
EXACT=PARENT+'/new_exact.json'
EXACT_SHA='7f253bef7645f3234fab9589db7871dca7079d87e48ea45c0f2af0139a34de4f'
EXACT_CACHE_SHA='74d2660dadbdbaf43667158ae86120f5637aac296c10302989b65698aecc2b28'
SPACE_SHA='d6490f5c46bd8792609aedca9241532cfca3dd65ddfe908ecbbb126e618ae306'
SPACE_MANIFEST_SHA='bfea798a14be25fb5e318dc5fbe3102ce463019b889ade8e16f74515210750f7'
ACTIVE=44400414
STATE=177814584
ADDED=0
LR=3e-5
BASE_LR=1e-5
LOCKS={0:'.course_directions_gpu0.lock'}
OPTIONS=dict(enabled=True,max_queries=16,weight=.5,no_noise=False,point_noise_ratio=.2)
NEW=('lr_spec.py','lr_contract.py','lr_worker.py','launch_lr.py','tests/test_lr.py')
def job(smoke=False):
    return dict(id=JOB,seed=20261013,output_rel='runs/'+JOB,flags=dict(multiscale_depth=True,reliability_gate=True,course_copy_paste=True,functional_smoke=smoke),max_job_seconds=50400)
def lease_end(now,used,absolute,limit):
    hard=min(absolute,DEADLINE,now+min(limit,GPU_MAX-used))
    if hard-now<=60:raise TimeoutError('No remaining immutable LR budget')
    return hard,hard-20
def arm_spec(root=None,arm=None):
    return SimpleNamespace(**{k:v for k,v in globals().items() if k.isupper()},job=job,lease_end=lease_end)
