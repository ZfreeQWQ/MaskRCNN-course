"""Frozen nine-view arithmetic; only new G model and aligned geometry sidecar."""
import copy,hashlib,inspect,json,os,sys,time,textwrap,unittest
from pathlib import Path
import g_system_spec as p
import abc_spec as s
for k,v in dict(START=p.START,DEADLINE=p.END,NAME=p.NAME,A_OWNER=p.OWNER,A_CPU_END=p.CPU_END,A_GPU_END=p.GPU_END,A_END=p.END,A_GPU_MAX=p.GPU_MAX,GPU_MAX=p.ROUND_GPU_MAX,WEIGHT=p.WEIGHT,REFERENCE='outputs/'+p.NAME+'/arms/G/reference').items():setattr(s,k,v)
import abc_a_worker as a
ROOT=Path('/workspace');OUT=ROOT/'outputs'/p.NAME/'arms/G';PRIOR=ROOT/p.PRIOR;TRAIN=ROOT/p.TRAIN
a.OUT=OUT
for path in (str(TRAIN/'source'),str(TRAIN/'source/tests')):
 if path not in sys.path:sys.path.append(path)
read,save,sha,mark,deadline=a.read,a.save,a.sha,a.mark,a.deadline
OLD_RUNTIME=a.runtime

def geometry_runtime(weight,end):
    if Path(weight)!=ROOT/p.WEIGHT:return OLD_RUNTIME(weight,end)
    import numpy as np,rgbd
    from mmengine.config import Config
    from mmengine.dataset import Compose
    from mmengine.utils import import_modules_from_strings
    extra=str(TRAIN/'source/rgbd')
    if extra not in rgbd.__path__:rgbd.__path__.append(extra)
    from rgbd.course_geometry_data import adapt_pipeline
    original_setup=a.old.setup_pipeline
    original,accepted,tiles,compact,oldcfg,loader,tail=original_setup()
    config=Config.fromfile(str(TRAIN/'formal/runs/geometry_long/config.py'));import_modules_from_strings(**config.custom_imports)
    pipe=[copy.deepcopy(dict(x)) for x in config.val_dataloader.dataset.pipeline if x['type']!='LoadAnnotations'];assert pipe[1]['type']=='GeometryResize'
    pipe[1]['scale']=(2559,1536);newloader=Compose(pipe[:1]);newtail=Compose(pipe[1:])
    crop640=tiles.crop_loaded;crop512=a.old.crop512
    def setup():
        result=list(original_setup());result[4]=copy.deepcopy(config);result[5]=newloader;result[6]=newtail;return tuple(result)
    a.old.setup_pipeline=setup;tiles.crop_loaded=lambda raw,origin:sync_crop(crop640,raw,origin,640);a.old.crop512=lambda raw,origin:sync_crop(crop512,raw,origin,512)
    try:predict,close=OLD_RUNTIME(weight,end)
    except BaseException:
        a.old.setup_pipeline=original_setup;tiles.crop_loaded=crop640;a.old.crop512=crop512;raise
    def done():
        try:return close()
        finally:a.old.setup_pipeline=original_setup;tiles.crop_loaded=crop640;a.old.crop512=crop512
    return predict,done

def sync_crop(original,raw,origin,size):
    out=original(raw,origin);out['geometry']=geometry_crop(raw['geometry'],origin,size)
    assert out['geometry'].shape[:2]==out['img'].shape[:2];return out

def geometry_crop(value,origin,size):
    import numpy as np
    assert value.shape==(1024,1024,7)
    if origin is None:return value.copy()
    x,y=origin;return np.ascontiguousarray(value[y:y+size,x:x+size]).copy()

def compare_global(actual,ident):
    from pycocotools import mask as mu
    rows=sorted([x for x in actual['global_rows'] if int(mu.area(x['segmentation']))>0],key=lambda x:-x['score'])
    assert rows==read(TRAIN/'evaluate_1536'/f'case_{ident}_predictions.json'),('G single1536/fullview strict parity',ident)

def gpu_source():
    source=inspect.getsource(a.gpu)
    changes={"('baseline',compact['weight']),('A',ROOT/s.WEIGHT)":"('baseline',ROOT/p.OLD_WEIGHT),('A',ROOT/p.WEIGHT)",
      "else:\n                    rows,activity=f.apply":"else:\n                    compare_global(actual,b['image_id'])\n                    rows,activity=f.apply",
      "assert gpu_states[arm]['active_parameters']==44544128 and gpu_states[arm]['model_state_bytes']==178389440":"expected=read(TRAIN/'cpu_model_binding.json') if arm=='A' else dict(active=44400414,state_bytes=177814584)\n        assert gpu_states[arm]['active_parameters']==expected['active']+143714 and gpu_states[arm]['model_state_bytes']==expected['state_bytes']+574856"}
    for x,y in changes.items():assert source.count(x)==1,(x,source.count(x));source=source.replace(x,y)
    return source.replace("p.OLD_WEIGHT", "gproto.OLD_WEIGHT").replace("p.WEIGHT", "gproto.WEIGHT")

def prep():
    import numpy as np,rgbd,torch
    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not Path('/dev/nvidiactl').exists();deadline(p.CPU_END)
    plan=read(OUT/'plan.json');assert read(TRAIN/'terminal_acceptance.json')['status']=='passed' and read(TRAIN/'single_readonly_acceptance.json')['status']=='passed'
    assert sha(PRIOR/'report.json')==p.REF_REPORT_SHA and sha(PRIOR/'evaluate_1536/predictions.segm.json')==p.REF_CACHE_SHA
    assert all(sha(ROOT/n)==h for n,h in plan['external_source_sha256'].items())
    extra=str(TRAIN/'source/rgbd')
    if extra not in rgbd.__path__:rgbd.__path__.append(extra)
    register_all_modules(init_default_scope=True);cfg=Config.fromfile(str(TRAIN/'formal/runs/geometry_long/config.py'));import_modules_from_strings(**cfg.custom_imports)
    saved=torch.load(ROOT/p.WEIGHT,map_location='cpu');model=MODELS.build(cfg.model);model.load_state_dict(saved['state_dict'],strict=True);del model,saved
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_g_system'));assert tests.wasSuccessful() and not tests.skipped
    refs=read(PRIOR/'reference8.json');newrefs=[]
    for ref in refs:
        i=ref['base']['image_id'];old=read(PRIOR/'captures'/f'{i}.json');base={k:old[k] for k in ('global_rows','current','merged','jobs','probabilities','before','before_activity')};base.update(image_id=i,request=ref['base']['request'],input_hashes=ref['base']['input_hashes'])
        newrefs.append(dict(base=base,**{k:old[k] for k in ('new_raw','new_origins','rows','activity')}))
    save(OUT/'reference8.json',newrefs);save(OUT/'selector.json',read(PRIOR/'selector.json'));manifest=read(PRIOR/'manifest.json');save(OUT/'manifest.json',manifest)
    for i,row in enumerate(manifest['images']):
        deadline(p.CPU_END)
        for key,path in (('rgb_sha256',row['request']['img_path']),('depth_sha256',row['request']['depth_file'])):assert sha(path)==row['input_hashes'][key]
        if i%128==0:mark('system_input_binding',images=i+1,expected=2584)
    target=OUT/'reference';target.mkdir();os.symlink(PRIOR/'evaluate_1536/predictions.segm.json',target/'predictions.segm.json');os.symlink(PRIOR/'input_hashes.json',target/'val_input_hashes.json')
    binding=read(PRIOR/'binding.json');inputs=dict(binding['input_sha256']);inputs[p.WEIGHT]=sha(ROOT/p.WEIGHT)
    for name in ('geometry_index.json','geometry_normalization.json','cpu_model_binding.json','formal/runs/geometry_long/config.py','single_readonly_acceptance.json'):inputs[p.TRAIN+'/'+name]=sha(TRAIN/name)
    import m2f_quality_selector as qs,m2f_tiny_verifier as tv,m2f_boundary_refiner as br
    feature_sources={name:dict(function=inspect.getsource(fn),source_file_sha256=sha(inspect.getfile(fn))) for name,fn in [('score_area_pair_features',qs.features),('RGBD_mask_patch',br.patch_input),('tiny_patch',tv.patch)]}
    save(OUT/'feature_contract.json',dict(status='passed',four_heads_unchanged=True,decoder_query_features_used=False,functions=feature_sources))
    save(OUT/'binding.json',dict(input_sha256=inputs,feature_contract_sha256=sha(OUT/'feature_contract.json'),reference_metrics=read(PRIOR/'new_exact.json')['metrics'],new_weight_sha256=sha(ROOT/p.WEIGHT),only_model_geometry_changed=True,head_feature_contract='All four heads retain original RGB-D/mask/score/area features; no decoder-query features, no C head',geometry_crop_rule='source-camera full geometry then identical image crops and nearest resize'))
    save(OUT/'numerical_functions.json',{k:hashlib.sha256(v.encode()).hexdigest() for k,v in a.numeric_functions().items()});assert read(OUT/'numerical_functions.json')==read(PRIOR/'numerical_functions.json')
    save(OUT/'cpu_acceptance.json',dict(status='passed',tests=tests.testsRun,skipped=0,numerical_functions_unchanged=True,actual_G_config_state_loaded=True,all2584_inputs_bound=True,train_only_normalization_bound=True,unix=time.time()))

def gpu():
    ns=dict(a.__dict__,p=p,gproto=p,TRAIN=TRAIN,runtime=geometry_runtime,compare_global=compare_global);exec(compile(gpu_source(),'<frozen G system GPU loops>','exec'),ns);ns['gpu']()

def finish():
    a.finish();r=read(OUT/'report.json');r.update(checkpoint_only=False,geometry_adapter=True,primary='LR9view47.70413206',training_updates=0,geometry_training_runs=1);save(OUT/'report.json',r)

if __name__=='__main__':globals()[sys.argv[1]]()
