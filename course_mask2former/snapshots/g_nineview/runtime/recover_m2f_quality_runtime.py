"""Directory-only bounded recovery; frozen numerical worker unchanged."""
import shutil,unittest,time
from pathlib import Path
import run_m2f_quality_runtime as r
START,PREP_END,GPU_END,ANALYSIS_END,DEADLINE=r.START,r.PREP_END,r.GPU_END,r.ANALYSIS_END,r.DEADLINE
NAME='course_m2f_quality_runtime_20261021_v2'
OWNER='21c4bb32ce2e48648ebd6a21c1638f19'
ROOT=Path('/workspace');OUT=ROOT/'outputs'/NAME;PARENT=ROOT/'outputs/course_m2f_quality_runtime_20261021_v1'
read,save,sha=r.read,r.save,r.sha
r.NAME,r.OWNER,r.OUT=NAME,OWNER,OUT
def prepare_output(path):
    (path/'actual8').mkdir()
class Contracts(unittest.TestCase):
    def test_output_parent_preexists(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);prepare_output(p);save(p/'actual8/1.json',{'x':1});self.assertEqual(read(p/'actual8/1.json'),{'x':1})
    def test_no_budget_reset(self):
        self.assertEqual((START,DEADLINE),(1792574220,1792576020))
def prep():
    r.cutoff(PREP_END)
    import os
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not Path('/dev/nvidiactl').exists()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    assert result.wasSuccessful() and not result.skipped
    assert read(PARENT/'status.json')['status']=='failed' and read(PARENT/'first_error.json')['phase']=='gpu'
    old=read(PARENT/'cpu_acceptance.json');assert old['status']=='passed' and old['tests']==35 and old['skipped']==0
    lease=read(PARENT/'gpu_released.json');assert lease['original_lock_released'] and 0<lease['seconds']<100
    for n,h in read(PARENT/'plan.json')['source_sha256'].items():assert sha(PARENT/'source'/n)==sha(OUT/'source'/n)==h
    for n in ('reference8.json','selector.json','binding.json'):shutil.copy2(PARENT/n,OUT/n)
    assert sha(OUT/'reference8.json')==old['reference8_sha256'] and sha(OUT/'selector.json')==old['selector_sha256']
    prepare_output(OUT)
    save(OUT/'cpu_acceptance.json',dict(old,tests=2,reused_prior_tests=35,source_unchanged=True,output_directory_verified=True,unix=time.time()))
    save(OUT/'recovery_binding.json',dict(prior_first_error_sha256=sha(PARENT/'first_error.json'),prior_release_sha256=sha(PARENT/'gpu_released.json'),prior_gpu_seconds=lease['seconds'],new_GPU_max_seconds=600-lease['seconds'],no_budget_reset=True,numerical_worker_unchanged=True))
def gpu():r.gpu()
def finish():
    assert read(PARENT/'gpu_released.json')['seconds']+read(OUT/'gpu_released.json')['seconds']<=600
    r.finish()
    report=read(OUT/'report.json');report['cumulative_gpu_seconds']=read(PARENT/'gpu_released.json')['seconds']+read(OUT/'gpu_released.json')['seconds']
    report['prior_failure_preserved']=True;save(OUT/'report.json',report)
if __name__=='__main__':
    import sys
    {'prep':prep,'gpu':gpu,'finish':finish}[sys.argv[1]]()
