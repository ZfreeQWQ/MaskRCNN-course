"""Reuse accepted bounded B supervisor, with one explicit LR contract."""
import inspect,json,os,shutil,subprocess,sys,time
from pathlib import Path
import lr_spec as spec
ROOT=Path('/mnt/nfs/fenghaibin/AI_Class');OUT=ROOT/'outputs'/spec.NAME
sys.path.insert(0,str(ROOT/spec.PARENT/'source'))
import abc_b_launcher as parent
from launch_nyu_sweep import atomic_json as save,sha256 as sha
q=parent.q
def read(p):return json.loads(Path(p).read_text())
def worker_source():
    source=parent.worker_source()
    changes={"'ai-m2f-match-'":"'ai-m2f-lr-'","'abc_b_worker.py',mode":"'lr_worker.py',mode",
             "assert r['config_checks']['tiny_matching_only']":"assert r['config_checks']['learning_rate_only']\n        lr=read(run/'lr_audit.json');assert lr['status']=='passed' and lr['base_lr']==s.LR and lr['iterations']==n and r['args']['lr']==s.LR"}
    for old,new in changes.items():assert source.count(old)==1,old;source=source.replace(old,new)
    return source
def start(commit):
    assert time.time()<spec.CPU_END and not OUT.exists()
    assert len(commit)==40 and all(c in '0123456789abcdef' for c in commit)
    previous=ROOT/spec.PARENT;p=read(previous/'plan.json')
    assert read(previous/'status.json')['status']=='complete' and read(previous/'terminal_acceptance.json')['status']=='passed'
    assert len(p['source_sha256'])==117 and all(sha(previous/'source'/n)==h for n,h in p['source_sha256'].items())
    for owner in (p['owner_token'],spec.OWNER):
        assert not subprocess.check_output(['docker','ps','-a','--filter','label=ai-course-directions-owner='+owner,'--format','{{.ID}}'],text=True).strip()
    provenance,reference=q.verify_inputs(ROOT);assert shutil.disk_usage(ROOT).free>=18*2**30
    assert sha(ROOT/spec.EXACT)==spec.EXACT_SHA and sha(ROOT/spec.REFERENCE/'predictions.segm.json')==spec.EXACT_CACHE_SHA
    worker_source();OUT.mkdir();(OUT/'logs').mkdir();shutil.copytree(previous/'source',OUT/'source')
    for n in spec.NEW:
        dst=OUT/'source'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(__file__).parent/n,dst)
    files={n:sha(OUT/'source'/n) for n in (*p['source_sha256'],*spec.NEW)}
    assert len(files)==122 and all(files[n]==h for n,h in p['source_sha256'].items())
    image=subprocess.check_output(['docker','image','inspect',q.IMAGE,'--format','{{.Id}}'],text=True).strip()
    plan=dict(protocol='B_LR_3e-5_only_30k',source_commit=commit,source_sha256=files,inherited_sha256=p['source_sha256'],
        workspace=str(ROOT),output=str(OUT),image_id=image,expected_provenance=provenance,reference_baseline=reference,
        owner_token=spec.OWNER,start_unix=spec.START,cpu_end=spec.CPU_END,smoke_end=spec.SMOKE_END,formal_end=spec.FORMAL_END,
        eval_end=spec.EVAL_END,deadline_unix=spec.DEADLINE,gpu_max_seconds=spec.GPU_MAX,selected_gpu=0,lock_wait_seconds=0,
        steps=30000,maximum_formal_runs=1,CNN_paused=True,CPU_cores=8,OMP_MKL_threads=2,no_extension=True,
        base_lr=spec.LR,baseline_lr=spec.BASE_LR,primary_reference=spec.REFERENCE,primary_reference_sha256=spec.REFERENCE_SHA,
        exact_cache_sha256=spec.EXACT_CACHE_SHA,no_submission=True,no_automatic_followup=True)
    save(OUT/'plan.json',plan)
    with (OUT/'supervisor.log').open('xb',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u',str(OUT/'source/launch_lr.py'),'worker'],stdin=subprocess.DEVNULL,
             stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    save(OUT/'launch.json',dict(pid=child.pid,unix=time.time(),plan_sha256=sha(OUT/'plan.json')))
    print(json.dumps(read(OUT/'launch.json')))
def worker():
    q.BASE_RECIPE=dict(q.BASE_RECIPE,lr=spec.LR)
    ns=dict(parent.prior.__dict__,spec=spec,ROOT=ROOT,OUT=OUT,s=None,q=q)
    exec(compile(worker_source(),'<bounded LR supervisor>','exec'),ns)
    ns['worker']()
if __name__=='__main__':start(sys.argv[2]) if sys.argv[1]=='start' else worker()
