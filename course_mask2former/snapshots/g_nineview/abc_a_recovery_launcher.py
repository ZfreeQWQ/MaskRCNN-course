"""Restore accepted dependency PYTHONPATH, same budget and same selected GPU0."""
import inspect,json,os,shutil,subprocess,sys,time
from pathlib import Path
import abc_spec as s
import abc_a_launcher as l
PRIOR=l.OUT;OUT=l.BATCH/'arms/A_recovery1';OWNER='7f728d93869445c6bd924367569fe2431'
NEW=('abc_a_recovery.py','abc_a_recovery_launcher.py')
EXTERNAL=('outputs/course_m2f_tiles_20261019_v1/source','outputs/course_m2f_geometry_20261020_v1/source')

def transformed():
    src=inspect.getsource(l.worker);changes={
        "used=0.;leases=[];selected=None":"used=prior_used;leases=prior_leases.copy();selected=0",
        'for card in (0,1):':'for card in (0,):',
        "rel='/workspace/outputs/'+s.NAME+'/arms/A'":"rel='/workspace/outputs/'+s.NAME+'/arms/A_recovery1'",
        "rel+'/source/runtime:/workspace/mmdetection'":"rel+'/source/runtime:/workspace/outputs/course_m2f_tiles_20261019_v1/source:/workspace/outputs/course_m2f_geometry_20261020_v1/source:/workspace/mmdetection'",
        "'abc_a_worker.py',mode":"'abc_a_recovery.py',mode",
    }
    for old,new in changes.items():assert src.count(old)==1,old;src=src.replace(old,new)
    return src

def start():
    assert time.time()<s.A_CPU_END and not OUT.exists()
    plan=l.read(PRIOR/'plan.json');release=l.read(PRIOR/'gpu_released.json')
    assert l.read(PRIOR/'status.json')['status']=='failed' and release['original_lock_released']
    assert 0<release['total_seconds']<60
    for owner in (s.A_OWNER,OWNER):assert not subprocess.check_output(['docker','ps','-a','--filter','label=ai-course-directions-owner='+owner,'--format','{{.ID}}'],text=True).strip()
    for n,h in plan['source_sha256'].items():assert l.sha(PRIOR/'source'/n)==h
    external={}
    for rel in EXTERNAL:
        directory=l.ROOT/rel;p=l.read(directory.parent/'plan.json')
        for n,h in p['source_sha256'].items():
            assert l.sha(directory/n)==h;external[str((directory/n).relative_to(l.ROOT))]=h
    transformed() # Validate exact edits before launching anything.
    (OUT/'logs').mkdir(parents=True);shutil.copytree(PRIOR/'source',OUT/'source')
    for n in NEW:shutil.copy2(Path(__file__).parent/n,OUT/'source'/n)
    files=dict(plan['source_sha256'],**{n:l.sha(OUT/'source'/n) for n in NEW})
    p=dict(plan,source_sha256=files,owner=OWNER,selected_gpu=0,prior_gpu_seconds=release['total_seconds'],prior_error_sha256=l.sha(PRIOR/'first_error.json'),prior_plan_sha256=l.sha(PRIOR/'plan.json'),external_source_sha256=external,recovery='accepted_dependency_search_path_only')
    l.save(OUT/'plan.json',p)
    with (OUT/'supervisor.log').open('xb',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u',str(OUT/'source/abc_a_recovery_launcher.py'),'worker'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    l.save(OUT/'launch.json',dict(pid=child.pid,unix=time.time(),plan_sha256=l.sha(OUT/'plan.json')))
    b=l.read(l.BATCH/'status.json');b['current']='A';b['arms']['A']='recovery1_running';b['A_active_root']=str(OUT);l.save(l.BATCH/'status.json',b)
    print(json.dumps(l.read(OUT/'launch.json')))

def worker():
    p=l.read(OUT/'plan.json');release=l.read(PRIOR/'gpu_released.json')
    assert p['prior_gpu_seconds']==release['total_seconds'] and p['prior_error_sha256']==l.sha(PRIOR/'first_error.json')
    s.A_OWNER=OWNER
    namespace=dict(l.__dict__,OUT=OUT,prior_used=release['total_seconds'],prior_leases=release['leases'])
    exec(compile(transformed(),'<fixed dependency recovery>','exec'),namespace);namespace['worker']()
if __name__=='__main__':start() if sys.argv[1]=='start' else worker()
