"""Second layer of the same G authorization; never a new budget or training."""
from pathlib import Path
import json
ROOT=Path('/workspace') if Path('/workspace/outputs').exists() else Path('/mnt/nfs/fenghaibin/AI_Class')
TRAIN='outputs/course_m2f_G_20261027_recovery2'
PRIOR='outputs/course_m2f_lr_system_20261027_v1/arms/LR_recovery3'
NAME='course_m2f_G_system_20261027_recovery2'
ARM='G';OWNER='c61fc4fe4e0b47ff982d8b583e5aa263'
START=1793068374;CPU_END=1793154774;GPU_END=1793172774;END=1793176374
GPU_MAX=14400;ROUND_GPU_MAX=86400
WEIGHT=TRAIN+'/formal/runs/geometry_long/iter_30000.pth'
OLD_WEIGHT='outputs/course_m2f_lr_20261026_v1/formal/runs/lr3e5_long/iter_30000.pth'
OLD_WEIGHT_SHA='b3a2e8e4420ff397e331d31e1ff58bcaae0ceddfb33794a09872ad27c09ccbb3'
REF_REPORT_SHA='dd49a789223456516c5b339760c6fb57319e5abe9cb26c216245d3d03e232f6f'
REF_CACHE_SHA='da6b5d3cef493709ae2abbc6f128dc914be795174f08f72c41cdbe83174bd239'
IMAGE='sha256:eff8b304fe898a9509ee7b6df88b17446519ec29c85d007c6f6ac709787e8140'
HOSTNAME='m6-3-a800';UUID='GPU-3c522036-79d8-7200-09b3-a394915e62df'
PROJECT_LOCK='.course_directions_gpu0.lock';UUID_LOCK='.course_gpu7_'+UUID+'.lock'
PRIVATE_OBSERVED_LOCK='/tmp/sparsekg_sm80_gpu_benchmark.lock'
NEW=('g_system_spec.py','g_system_worker.py','g_system_launcher.py','test_g_system.py')

PREVIOUS_GPU_SECONDS=8.311664342880249
