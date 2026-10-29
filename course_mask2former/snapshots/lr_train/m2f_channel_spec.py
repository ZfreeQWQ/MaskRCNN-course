"""One approved channel-calibration arm. All deadlines include preparation."""
START=1792743562  # 2026-10-23 16:19:22 Asia/Shanghai
CPU_END=START+2700
SMOKE_END=START+3600
FORMAL_END=START+37800
EVAL_END=START+40500
DEADLINE=START+43200
GPU_MAX=37800
NAME='course_m2f_channel_20261023_v2'
JOB='channel_long'
OWNER='c85c708cc78f4b7bacbd160cefe5e9dc'
REFERENCE='outputs/course_m2f_copypaste1536_20261018_v1/evaluate_1536'
REFERENCE_SHA='b4e4c3dc6ebfbe3a63e88be36be6065388d63987e92323f7dbde0e57100451ee'
CP_RUN='outputs/course_directions_20261015_v1/formal/runs/copy_paste_long'
CP_REPORT='790a771d893437dba23e41180efbf149be4cec5398eab733ad924e2d23d14394'
CP_WEIGHT='bdea9c0f87864941e9ba34abceb561328fc4a950051ea4e0ad59cf81219ccd1a'
NEW=('m2f_channel_spec.py','m2f_channel_worker.py','launch_m2f_channel.py',
     'tests/test_m2f_channel.py','rgbd/course_channel.py')
ACTIVE=44588638
STATE=178567480
ADDED=188224
LOCKS={0:'.course_directions_gpu0.lock',1:'.course_maskrcnn_gpu1_formal.lock'}

def job(smoke=False):
    return dict(id=JOB,seed=20261013,output_rel='runs/'+JOB,
        flags=dict(multiscale_depth=True,reliability_gate=True,course_copy_paste=True,functional_smoke=smoke),
        max_job_seconds=34200)

def lease_end(now,used,absolute,limit):
    hard=min(absolute,now+min(limit,GPU_MAX-used))
    if hard-now<=60:raise TimeoutError('Insufficient immutable GPU budget')
    return hard,hard-20
