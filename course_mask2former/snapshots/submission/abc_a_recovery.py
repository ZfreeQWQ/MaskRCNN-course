"""Dependency search-path-only recovery, preserves old frozen numerical worker."""
import json,os,shutil,sys,time
from pathlib import Path
import abc_spec as s
import abc_a_worker as a
ROOT=Path('/workspace');PRIOR=ROOT/'outputs'/s.NAME/'arms/A';OUT=ROOT/'outputs'/s.NAME/'arms/A_recovery1'
a.OUT=OUT

def prep():
    a.deadline(s.A_CPU_END);assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not Path('/dev/nvidiactl').exists()
    p=a.read(PRIOR/'plan.json');assert a.read(PRIOR/'status.json')['status']=='failed'
    assert a.read(PRIOR/'first_error.json')['phase']=='gpu'
    assert 'ModuleNotFoundError: No module named' in (PRIOR/'logs/gpu.log').read_text()
    assert not (PRIOR/'gpu_report.json').exists() and not (PRIOR/'captures').exists()
    for n,h in p['source_sha256'].items():assert a.sha(PRIOR/'source'/n)==a.sha(OUT/'source'/n)==h,n
    cpu=a.read(PRIOR/'cpu_acceptance.json');assert cpu['status']=='passed' and cpu['tests']==42
    for n in ('reference8.json','selector.json','manifest.json','binding.json','numerical_functions.json'):shutil.copy2(PRIOR/n,OUT/n)
    plan=a.read(OUT/'plan.json')
    for n,h in plan['external_source_sha256'].items():assert a.sha(ROOT/n)==h
    # Exercise precisely the previously missing import/config/pipeline setup on CPU.
    import m2f_tiles,m2f_geometry_compact_check
    a.old.START,a.old.DEADLINE=s.START,s.A_END
    original,accepted,tiles,compact,cfg,loader,tail=a.old.setup_pipeline()
    assert Path(tiles.__file__).resolve()==(ROOT/'outputs/course_m2f_tiles_20261019_v1/source/m2f_tiles.py').resolve()
    assert callable(loader) and callable(tail)
    a.save(OUT/'cpu_acceptance.json',dict(status='passed',tests=1,reused_prior_tests=42,skipped=0,
        import_pipeline_cpu_verified=True,numerical_sources_unchanged=True,prior_cpu_sha256=a.sha(PRIOR/'cpu_acceptance.json'),prior_first_error_sha256=a.sha(PRIOR/'first_error.json'),unix=time.time()))

if __name__=='__main__':
    mode=sys.argv[1]
    prep() if mode=='prep' else getattr(a,mode)()
