"""Approved finite serial A/B/C, no silent renewal."""
from pathlib import Path
START=1792894252
DEADLINE=1793002252
GPU_MAX=82800
NAME='course_m2f_abc_20261025_v1'
A_OWNER='644cc2472ad44b079538f84117d59c5d1'
A_END=START+21600
A_CPU_END=START+3600
A_GPU_END=A_END-1800
A_GPU_MAX=18000
RUNTIME='outputs/course_m2f_quality_runtime_20261021_v2'
REFERENCE='outputs/course_m2f_fine512_quality_fullval_20261021_v1'
NOISE='outputs/course_m2f_pointnoise_20261024_v1'
WEIGHT=NOISE+'/formal/runs/pointnoise_long/iter_30000.pth'
WEIGHT_SHA='e53ea799348ae53403d15fba07c7d763143f07879191778142a63c2002766ece'
REF_SHA='cf78ffd816bd06662b126f552f665979a343fb084b11ea03432a131e4fb79d41'
LOCKS={0:'.course_directions_gpu0.lock',1:'.course_maskrcnn_gpu1_formal.lock'}
IMAGE='sha256:eb5bed0d49efd109b91d5365131cc7f25391b8695347086d2e3dd12953d7702c'
NEW=('abc_spec.py','abc_a_worker.py','abc_a_launcher.py','test_abc_a.py','accepted_error_analysis.py')
