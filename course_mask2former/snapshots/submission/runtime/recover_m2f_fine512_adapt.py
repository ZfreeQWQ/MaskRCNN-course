"""Repair only fit M2F backend flags; reuse completed fresh128 without rerun."""
import json,os,shutil,time,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import run_m2f_fine512_adapt as base
from run_m2f_fine512_adapt import START,PREP_END,CAPTURE_END,LABEL_END,TRAIN_END,ANALYSIS_END,DEADLINE,read,save,sha
NAME='course_m2f_fine512_adapt_20261021_v2';OWNER='2132c22ea9274fb38f5a0d82c09ebd50'
ROOT=Path('/workspace');OUT=ROOT/'outputs'/NAME;PRIOR=ROOT/'outputs/course_m2f_fine512_adapt_20261021_v1'
base.OUT=base.old.OUT=OUT

def fit_flags(torch):
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.deterministic=False
    return torch

class Contracts(unittest.TestCase):
    def test_exact_flags(self):
        t=NS(backends=NS(cudnn=NS(allow_tf32=False,deterministic=True,benchmark=False),cuda=NS(matmul=NS(allow_tf32=False))))
        self.assertIs(fit_flags(t),t);self.assertTrue(t.backends.cudnn.allow_tf32);self.assertFalse(t.backends.cudnn.deterministic)
        self.assertFalse(t.backends.cuda.matmul.allow_tf32);self.assertFalse(t.backends.cudnn.benchmark)
    def test_original_deadlines(self):
        self.assertEqual(START,1792528506);self.assertEqual(DEADLINE,1792535706);self.assertEqual(CAPTURE_END,1792531806)

def prep():
    assert time.time()<CAPTURE_END and not Path('/dev/nvidiactl').exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts));assert tests.wasSuccessful() and not tests.skipped
    prior=read(PRIOR/'plan.json');assert prior['deadline_unix']==DEADLINE
    assert read(PRIOR/'status.json')['status']=='failed' and read(PRIOR/'gpu_leases.json')['original_lock_released']
    for n,h in prior['source_sha256'].items():assert sha(OUT/'source'/n)==h,n
    cpu=read(PRIOR/'cpu_acceptance.json');assert cpu['status']=='passed' and cpu['tests']==22 and cpu['unix']<PREP_END
    c=read(PRIOR/'capture_report.json');assert c['status']=='passed' and c['calls']==dict(eval=1152,train=0) and c['state_before']==c['state_after']
    (OUT/'captures').mkdir()
    for f in c['files']:
        p=PRIOR/'captures'/f"{f['image_id']}.json";assert sha(p)==f['sha256'];shutil.copy2(p,OUT/'captures'/p.name)
    for n in ('prepared_manifest.json','binding.json','cpu_acceptance.json','capture_report.json','throughput_gate.json'):
        shutil.copy2(PRIOR/n,OUT/n);assert sha(OUT/n)==sha(PRIOR/n)
    save(OUT/'recovery_acceptance.json',dict(status='passed',tests=tests.testsRun,skipped=0,prior_cpu_tests=22,prior_check_images=128,prior_check_eval_calls=1152,
        reuse_exact=True,prior_plan_sha256=sha(PRIOR/'plan.json'),prior_source_sha256=prior['source_sha256'],prior_gpu_leases=read(PRIOR/'gpu_leases.json'),
        prior_first_error=read(PRIOR/'first_error.json'),only_repair='fit M2F cudnn allow_tf32=True deterministic=False, matching existing inference',unix=time.time()))

def capture():
    original=base.w.base.configure_torch
    def configured():
        t=fit_flags(original());save(OUT/'fit_inference_flags.json',dict(cudnn_allow_tf32=t.backends.cudnn.allow_tf32,cudnn_deterministic=t.backends.cudnn.deterministic,matmul_allow_tf32=t.backends.cuda.matmul.allow_tf32,benchmark=t.backends.cudnn.benchmark,unix=time.time()));return t
    base.w.base.configure_torch=configured
    try:base.fit_capture()
    finally:base.w.base.configure_torch=original
    fit=read(OUT/'fit_capture_report.json');check=read(OUT/'capture_report.json')
    assert fit['calls']==dict(eval=1536,train=0) and check['calls']==dict(eval=1152,train=0)
    save(OUT/'combined_capture.json',dict(status='passed',eval_calls=2688,train_calls=0,check_reused=True,fit=sha(OUT/'fit_capture_report.json'),check=sha(OUT/'capture_report.json')))

def traincheck():
    base.traincheck()
    import torch
    ck=torch.load(OUT/'verifier_512_2000.pth',map_location='cpu');steps=[]
    for item in ck['optimizer']['state'].values():
        steps.append(int(item['step']))
        assert all(bool(torch.isfinite(v).all()) for v in item.values() if torch.is_tensor(v))
    assert set(steps)=={2000} and all(bool(torch.isfinite(v).all()) for v in ck['state_dict'].values())
    r=read(OUT/'train_report.json');r['optimizer_reloaded_cpu_finite']=True;save(OUT/'train_report.json',r)

if __name__=='__main__':
    import sys
    {'prep':prep,'capture':capture,'labels':base.make_patches,'traincheck':traincheck,'finish':base.finish}[sys.argv[1]]()
