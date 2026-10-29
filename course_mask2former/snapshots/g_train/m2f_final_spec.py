"""Final finite two-arm round. No automatic third experiment or budget renewal."""
import json,os
from pathlib import Path
from types import SimpleNamespace
import m2f_channel_spec as prior

START=1792746097  # 2026-10-23 17:01:37 Asia/Shanghai; includes implementation
PREP_END=START+7200
DEADLINE=START+36*3600
GPU_MAX=30*3600
NAME='course_m2f_final_round_20261023_v1'
ARMS=('channel','pilot')
OWNERS={'channel':'328da7c386534bfd94ba8cc8193c46ae','pilot':'74839e8bc7fc4315be62e81b1c3cad18'}
NEW=(*prior.NEW,'m2f_local_decoder.py','m2f_final_spec.py','m2f_final_worker.py',
     'launch_m2f_final.py','rgbd/course_pilot.py','tests/test_m2f_final.py')
EXACT='outputs/course_m2f_channel_20261023_v2/reference_exact.json'
EXACT_CACHE_SHA='b72e669f63384590c07b94819f1f154d3046a464d0e87438b4a71b1b5b9cc588'

def arm_spec(root,arm):
    assert arm in ARMS
    b=json.loads((Path(root)/'outputs'/NAME/'arms'/arm/'arm_budget.json').read_text())
    assert b['round_start']==START and b['round_deadline']==DEADLINE
    start=b['start'];end=min(start+16*3600,DEADLINE-3600)
    d={k:getattr(prior,k) for k in ('REFERENCE','REFERENCE_SHA','CP_RUN','CP_REPORT','CP_WEIGHT','LOCKS')}
    d.update(ARM=arm,NAME=NAME+'/arms/'+arm,JOB=arm+'_long',OWNER=OWNERS[arm],START=start,
        CPU_END=min(start+1800,PREP_END) if arm=='channel' else start+1800,
        SMOKE_END=start+2700,FORMAL_END=end-3600,EVAL_END=end-900,DEADLINE=end,
        GPU_MAX=15*3600,ACTIVE=44588638 if arm=='channel' else 44400414,
        STATE=178567480 if arm=='channel' else 177814584,ADDED=188224 if arm=='channel' else 0)
    s=SimpleNamespace(**d)
    s.job=lambda smoke=False:dict(id=s.JOB,seed=20261013,output_rel='runs/'+s.JOB,
        flags=dict(multiscale_depth=True,reliability_gate=True,course_copy_paste=True,functional_smoke=smoke),
        max_job_seconds=14*3600)
    def lease(now,used,absolute,limit):
        hard=min(absolute,now+min(limit,s.GPU_MAX-used))
        if hard-now<=60:raise TimeoutError('Insufficient immutable final-round GPU budget')
        return hard,hard-20
    s.lease_end=lease
    return s
