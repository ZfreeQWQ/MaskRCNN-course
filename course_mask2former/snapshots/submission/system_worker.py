"""Frozen A numerical runtime; substitute accepted B checkpoint only."""
import ast,copy,inspect,json,os,shutil,sys,time,unittest
from pathlib import Path
from types import SimpleNamespace
import system_spec as p
import abc_spec as s
# Only orchestration/reference parameters change, never inherited source files.
for key,value in dict(START=p.START,DEADLINE=p.DEADLINE,NAME=p.NAME,A_OWNER=p.OWNER,A_CPU_END=p.CPU_END,A_GPU_END=p.B_GPU_END,A_END=p.B_END,A_GPU_MAX=p.B_GPU_MAX,GPU_MAX=p.GPU_MAX,WEIGHT=p.WEIGHT,WEIGHT_SHA=p.WEIGHT_SHA,REFERENCE='outputs/'+p.NAME+'/arms/B_into_A/reference',REF_SHA=p.REF_SHA).items():setattr(s,key,value)
import abc_a_worker as a
ROOT=Path('/workspace');OUT=ROOT/'outputs'/p.NAME/'arms/B_into_A';PRIOR=ROOT/p.PRIOR
a.OUT=OUT

def assert_baseline(actual,ref):
    ident=ref['base']['image_id'];assert actual==a.read(PRIOR/'captures'/f'{ident}.json'),ident

def assert_new_global(actual,ident):
    from pycocotools import mask as mu
    rows=[r for r in actual['global_rows'] if int(mu.area(r['segmentation']))>0]
    rows=sorted(rows,key=lambda r:-r['score']);expected=a.read(ROOT/p.B/'evaluate_1536'/f'case_{ident}_predictions.json')
    assert rows==expected,('B global prediction differs from accepted standalone',ident)

def transformed(mode):
    source=inspect.getsource(getattr(a,mode));changes={}
    if mode=='gpu':
        changes={"('baseline',compact['weight'])":"('baseline',ROOT/p.BASE_WEIGHT)","r.assert_parity(actual,ref)":"assert_baseline(actual,ref)","r.assert_parity(actual,refs[0])":"assert_baseline(actual,refs[0])","'A'":"'B_into_A'","p=OUT/'captures'/f\"{b['image_id']}.json\";save(p,actual)":"assert_new_global(actual,b['image_id']);path=OUT/'captures'/f\"{b['image_id']}.json\";save(path,actual)"}
        # Local variable p would shadow the module spec throughout gpu; replace all path uses.
        source=source.replace("p=OUT/'captures'/f'{i}.json'","path=OUT/'captures'/f'{i}.json'").replace('save(p,actual)','save(path,actual)').replace('sha(p)','sha(path)')
        changes.pop("p=OUT/'captures'/f\"{b['image_id']}.json\";save(p,actual)")
        changes["p=OUT/'captures'/f\"{b['image_id']}.json\";save(path,actual)"]="assert_new_global(actual,b['image_id']);path=OUT/'captures'/f\"{b['image_id']}.json\";save(path,actual)"
    elif mode=='finish':changes={"ARM='A'":"ARM='B_into_A'"}
    else:raise ValueError(mode)
    for old,new in changes.items():
        assert old in source,(mode,old);source=source.replace(old,new)
    return source

def prep():
    import torch
    a.deadline(p.CPU_END);assert not Path('/dev/nvidiactl').exists() and os.environ['CUDA_VISIBLE_DEVICES']==''
    test=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_system'));assert test.wasSuccessful() and not test.skipped
    plan=a.read(OUT/'plan.json');priorplan=a.read(PRIOR/'plan.json');inputs={}
    for n,h in priorplan['source_sha256'].items():assert a.sha(PRIOR/'source'/n)==a.sha(OUT/'source'/n)==h,n
    for n,h in priorplan['external_source_sha256'].items():assert a.sha(ROOT/n)==h,n
    assert a.sha(PRIOR/'report.json')==p.REF_SHA and a.read(PRIOR/'report.json')['status']=='passed'
    for arm,expected in ((p.B,p.B_REPORT_SHA),(p.C,p.C_REPORT_SHA)):
        path=ROOT/arm/('evaluate_1536/report.json' if arm==p.B else 'report.json');assert a.sha(path)==expected and a.read(ROOT/arm/'status.json')['status']=='complete';inputs[str(path.relative_to(ROOT))]=expected
    assert a.sha(ROOT/p.B/'evaluate_1536/predictions.segm.json')==p.B_CACHE_SHA
    inputs[p.B+'/evaluate_1536/predictions.segm.json']=p.B_CACHE_SHA
    assert a.read(ROOT/p.B/'comparison.json')['delta_points']['APs']>0 and a.read(ROOT/p.C/'comparison.json')['delta_points']['APs']>0
    for rel in ('reference8.json','selector.json','manifest.json','numerical_functions.json'):shutil.copy2(PRIOR/rel,OUT/rel)
    ref=OUT/'reference';ref.mkdir()
    for name,source in {'predictions.segm.json':PRIOR/'evaluate_1536/predictions.segm.json','val_input_hashes.json':PRIOR/'input_hashes.json','full_val_metrics.json':PRIOR/'new_exact.json','report.json':PRIOR/'report.json'}.items():
        (ref/name).symlink_to(os.path.relpath(source,ref));inputs[str(source.relative_to(ROOT))]=a.sha(source)
    captured={x['image_id']:x['capture_sha256'] for x in a.read(PRIOR/'predictions_frozen.json')['files']}
    for refrow in a.read(OUT/'reference8.json'):
        ident=refrow['base']['image_id'];path=PRIOR/'captures'/f'{ident}.json';assert a.sha(path)==captured[ident];inputs[str(path.relative_to(ROOT))]=captured[ident]
        bcase=ROOT/p.B/'evaluate_1536'/f'case_{ident}_predictions.json';inputs[str(bcase.relative_to(ROOT))]=a.sha(bcase)
    exact=a.read(PRIOR/'new_exact.json');assert exact['AP_runs']==1 and abs(exact['metrics']['APs']*100-42.15970417060633)<1e-10
    assert exact['cache_sha256']==a.sha(PRIOR/'evaluate_1536/predictions.segm.json')
    manifest=a.read(OUT/'manifest.json');assert len(manifest['ids'])==2584 and manifest['ids']==[x['image_id'] for x in a.read(PRIOR/'input_hashes.json')]
    oldbinding=a.read(PRIOR/'binding.json')
    for name,h in oldbinding['input_sha256'].items():assert a.sha(ROOT/name)==h;inputs[name]=h
    assert a.sha(ROOT/p.BASE_WEIGHT)==p.BASE_WEIGHT_SHA and a.sha(ROOT/p.WEIGHT)==p.WEIGHT_SHA
    inputs[p.BASE_WEIGHT]=p.BASE_WEIGHT_SHA;inputs[p.WEIGHT]=p.WEIGHT_SHA
    old=torch.load(ROOT/p.BASE_WEIGHT,map_location='cpu')['state_dict'];new=torch.load(ROOT/p.WEIGHT,map_location='cpu')['state_dict']
    assert old.keys()==new.keys() and all(old[k].dtype==new[k].dtype and old[k].shape==new[k].shape for k in old)
    assert any(not torch.equal(old[k],new[k]) for k in old) and all(torch.isfinite(v).all() for v in new.values() if v.is_floating_point());del old,new
    a.old.START,a.old.DEADLINE=p.START,p.B_END
    original,accepted,tiles,compact,cfg,loader,tail=a.old.setup_pipeline();assert callable(loader) and callable(tail)
    assert Path(a.r.__file__).resolve().is_relative_to(OUT/'source/runtime')
    hashes={k:__import__('hashlib').sha256(v.encode()).hexdigest() for k,v in a.numeric_functions().items()};assert hashes==a.read(PRIOR/'numerical_functions.json')
    a.save(OUT/'binding.json',dict(input_sha256=inputs,reference_metrics=exact['metrics'],new_weight_sha256=p.WEIGHT_SHA,only_model_checkpoint_changed=True,baseline_actual_eight_root=str(PRIOR/'captures')))
    a.save(OUT/'adapter_sources.json',dict(gpu=transformed('gpu'),finish=transformed('finish')))
    a.deadline(p.CPU_END);a.save(OUT/'cpu_acceptance.json',dict(status='passed',tests=test.testsRun,skipped=0,inherited_files=len(priorplan['source_sha256']),four_numeric_functions_unchanged=True,actual_setup_pipeline=True,baseline_AP_runs=0,baseline_APs=exact['metrics']['APs']*100,changed_only_checkpoint=True,unix=time.time()))

def execute(mode):
    namespace=dict(a.__dict__,p=p,assert_baseline=assert_baseline,assert_new_global=assert_new_global)
    exec(compile(transformed(mode),'<system '+mode+'>','exec'),namespace);namespace[mode]()
    if mode=='finish':
        report=a.read(OUT/'report.json');report.update(protocol='B_into_A_system_acceptance',baseline='A_nineview_42.15970417',candidate='B_30k_in_A_frozen_nineviews',source_plan_sha256=a.sha(OUT/'plan.json'));a.save(OUT/'report.json',report)
if __name__=='__main__':prep() if sys.argv[1]=='prep' else execute(sys.argv[1])
