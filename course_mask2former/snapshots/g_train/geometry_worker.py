"""Bounded G experiment built on byte-identical accepted LR training sources."""
import copy,inspect,json,os,sys,time,subprocess
from pathlib import Path
import geometry_spec as spec
import lr_spec
for k,v in vars(spec).items():
    if k.isupper() or k in ('job','lease_end','arm_spec'):setattr(lr_spec,k,v)
import lr_worker as lr
w=lr.w;ROOT=lr.ROOT;OUT=lr.OUT;save=lr.save;read=lr.read;sha=lr.sha
old_build=lr.build

def capacity():
    if (OUT/'cpu_model_binding.json').exists():
        b=read(OUT/'cpu_model_binding.json')
        for o in (spec,lr.spec,w.s):o.ACTIVE=b['active'];o.STATE=b['state_bytes'];o.ADDED=b['added_parameters']

def build(args,cats):
    from rgbd.course_geometry_data import adapt_pipeline
    cfg=old_build(args,cats);cfg.custom_imports.imports+=['rgbd.course_geometry_data','rgbd.course_geometry_model']
    cfg.model.type='GeometryMatchedMask2Former';cfg.model.geometry_options=spec.GEO_OPTIONS.copy()
    norm=read(OUT/'geometry_normalization.json');cfg.model.data_preprocessor.update(type='GeometryDataPreprocessor',geometry_mean=norm['mean'],geometry_std=norm['std'])
    for name in ('train_dataloader','val_dataloader','test_dataloader'):
        if name in cfg and cfg[name] is not None:cfg[name].dataset.pipeline=adapt_pipeline(cfg[name].dataset.pipeline,str(OUT/'geometry_index.json'))
    cfg.optim_wrapper.paramwise_cfg.custom_keys['geometry_fusion']=dict(lr_mult=10.)
    cfg.custom_hooks.append(dict(type='GeometryAuditHook'))
    return cfg

def compare_recipe(cfg):
    from mmengine.config import Config
    from rgbd.course_geometry_data import adapt_pipeline
    assert sha(ROOT/spec.CP_RUN/'report.json')==spec.CP_REPORT
    old=Config.fromfile(str(ROOT/spec.CP_RUN/'config.py')).to_dict();new=cfg.to_dict()
    old['model']['type']='GeometryMatchedMask2Former';old['model']['geometry_options']=spec.GEO_OPTIONS.copy();norm=read(OUT/'geometry_normalization.json')
    old['model']['data_preprocessor'].update(type='GeometryDataPreprocessor',geometry_mean=norm['mean'],geometry_std=norm['std'])
    old['optim_wrapper']['paramwise_cfg']['custom_keys']['geometry_fusion']=dict(lr_mult=10.)
    for name in ('train_dataloader','val_dataloader'):old[name]['dataset']['pipeline']=adapt_pipeline(old[name]['dataset']['pipeline'],str(OUT/'geometry_index.json'))
    keys=('model','train_cfg','optim_wrapper','param_scheduler','randomness','train_dataloader','val_dataloader','val_evaluator','auto_scale_lr','env_cfg')
    differences=[k for k in keys if old[k]!=new[k]];assert not differences,differences
    return dict(status='passed',geometry_only=True,pilot_only=True,baseline_report_sha256=spec.CP_REPORT,compared_keys=keys)

original_initial=None
def initial_state(model,state,detail=False):
    import torch
    if not hasattr(model,'geometry_fusion'):return original_initial(model,state,detail)
    current=model.state_dict();missing=set(current)-set(state);allowed={k for k in current if k.startswith(('spatial_fusion.','geometry_fusion.'))}
    assert missing==allowed and not set(state)-set(current)
    current.update(state);model.load_state_dict(current,strict=True)
    assert all(not v.count_nonzero() for k,v in current.items() if k.startswith('geometry_fusion.projections.'))
    return {k:v.detach().cpu().clone() for k,v in current.items() if k.startswith(('spatial_fusion.','depth_adapter.','panoptic_head.cls_embed.','geometry_fusion.'))}

def patch_initial():
    global original_initial
    import train_course_directions as t
    if original_initial is None:original_initial=t.initial_state;t.initial_state=initial_state

def cpu():
    import torch
    from mmengine.optim import build_optim_wrapper
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    import train_course_directions as t
    import geometry_cpu
    from test_abc_b import actual_matching_check
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not Path('/dev/nvidiactl').exists();w.deadline(spec.CPU_END)
    # Isolate inherited tests from this wrapper's spec and trainer substitutions.
    suites=['test_geometry','test_lr','test_abc_b','test_m2f_pointnoise','test_m2f_final','test_m2f_channel','test_course_directions_training','test_course_copy_paste','test_course_feature_models','test_course_resolution']
    script="import unittest,json; r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames("+repr(suites)+")); assert r.wasSuccessful() and not r.skipped; print('CPU_TESTS_PASSED',r.testsRun)"
    subprocess.run([sys.executable,'-c',script],check=True,timeout=max(1,spec.CPU_END-time.time()))
    register_all_modules(init_default_scope=True);geometry_cpu.prepare(sys.modules[__name__]);args=lr.args_for();cats=[dict(id=1,name='component',supercategory='component')]
    oldcfg=old_build(args,cats);cfg=build(args,cats);checks=compare_recipe(cfg);import_modules_from_strings(**cfg.custom_imports)
    assert sha(args.checkpoint)==t.COCO_SHA;state=t.base.component_state(torch.load(args.checkpoint,map_location='cpu')['state_dict']);patch_initial()
    torch.manual_seed(args.seed);old=MODELS.build(oldcfg.model);old.init_weights();initial_state(old,state);rng=torch.get_rng_state().clone()
    torch.manual_seed(args.seed);new=MODELS.build(cfg.model);new.init_weights();initial_state(new,state)
    assert torch.equal(rng,torch.get_rng_state());a=old.state_dict();b=new.state_dict();assert all(torch.equal(v,b[k]) for k,v in a.items())
    added=set(b)-set(a);assert added and all(k.startswith('geometry_fusion.') for k in added)
    oldopt=build_optim_wrapper(old,oldcfg.optim_wrapper);newopt=build_optim_wrapper(new,cfg.optim_wrapper);og=lr.groups(old,oldopt);ng=lr.groups(new,newopt)
    assert all(ng[k]==v for k,v in og.items());assert all(abs(v['lr']-spec.LR*10)<1e-12 for k,v in ng.items() if k.startswith('geometry_fusion.'))
    save(OUT/'optimizer_group_binding.json',dict(status='passed',old=og,new=ng,common_groups_unchanged=True,new_geometry_lr_mult=10))
    del oldopt,newopt,state,a,b
    geometry_cpu.actual_checks(sys.modules[__name__],old,new,oldcfg,cfg)
    # The preceding full-loss probe already used matching; the inherited unit
    # probe requires a fresh counter. Reset only this CPU test object's audit.
    new.panoptic_head.match_activity=dict(calls=0,tiny_targets=0,replaced_points=0);new.panoptic_head._tiny_cache={}
    actual_matching_check(new.panoptic_head,OUT/'matching_cpu_correctness.json')
    cfg.dump(str(OUT/'formal_template.py'))
    save(OUT/'cpu_model_binding.json',dict(status='passed',common_tensors_equal=True,RNG_equal=True,added_parameters=sum(v.numel() for k,v in new.state_dict().items() if k in added),active=new.active_parameter_count(),state_bytes=sum(v.numel()*v.element_size() for v in new.state_dict().values()),COCO_sha256=t.COCO_SHA,recipe=checks,normalization_sha256=sha(OUT/'geometry_normalization.json'),index_sha256=sha(OUT/'geometry_index.json'),optimizer_groups_sha256=sha(OUT/'optimizer_group_binding.json')))
    del old,new;capacity()
    for path,digest in [(ROOT/spec.CP_RUN/'iter_30000.pth',spec.CP_WEIGHT),(ROOT/spec.REFERENCE/'report.json',spec.REFERENCE_SHA),(ROOT/spec.REFERENCE/'predictions.segm.json',spec.EXACT_CACHE_SHA),(ROOT/spec.EXACT,spec.EXACT_SHA)]:assert sha(path)==digest,str(path)
    exact=read(ROOT/spec.EXACT);assert exact['status']=='passed' and exact['AP_runs']==1
    save(OUT/'reference_exact.json',exact);save(OUT/'reference_reuse.json',dict(path=spec.EXACT,sha256=spec.EXACT_SHA,new_baseline_AP_runs=0))
    w.deadline(spec.CPU_END);save(OUT/'cpu_acceptance.json',dict(status='passed',test_suites=suites,zero_skip=True,binding_sha256=sha(OUT/'cpu_model_binding.json'),actual_input_parity=True,unix=time.time()))

def register_hook():
    import torch
    # Preserve inherited update/noise/matching/LR checks; enlarge fixed val allowance.
    src=inspect.getsource(lr.b.parent.old_register);assert src.count('6*1500+300')==1
    ns=dict(w.__dict__);exec(compile(src.replace('6*1500+300','6*2400+600'),'<geometry conservative fixed 200 hook>','exec'),ns)
    lr.b.parent.old_register=ns['register_hook'];lr.register_hook()
    from mmengine.hooks import Hook
    from mmengine.registry import HOOKS
    @HOOKS.register_module()
    class GeometryAuditHook(Hook):
        priority='VERY_LOW'
        def before_train(self,runner):
            self.initial={k:v.detach().cpu().clone() for k,v in runner.model.geometry_fusion.state_dict().items()};self.steps=0
            assert runner.model.geometry_activity==dict(train_calls=0,train_samples=0,dropped_samples=0,eval_calls=0)
        def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
            self.steps+=1
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in runner.model.geometry_fusion.parameters())
        def after_train(self,runner):
            state=runner.model.geometry_fusion.state_dict();updated={k:not torch.equal(v,state[k].detach().cpu()) for k,v in self.initial.items()};a=runner.model.geometry_activity
            assert all(updated.values()) and all(torch.isfinite(v).all() for v in state.values())
            assert a['train_calls']==runner.max_iters and a['train_samples']==2*runner.max_iters and 0<a['dropped_samples']<a['train_samples']
            save(Path(runner.work_dir)/'geometry_audit.json',dict(status='passed',iterations=self.steps,activity=a,updated=updated,geometry_index_sha256=sha(OUT/'geometry_index.json'),normalization_sha256=sha(OUT/'geometry_normalization.json'),inference_GT=False,dropout_RNG='independent_cpu_generator',normal_frame='source_camera'))

def evaluate():
    import course_resolution_probe as probe
    from rgbd.course_geometry_data import adapt_pipeline
    original=probe.scaled_pipeline
    def scaled(pipe,scale=1024):
        if pipe[0]['type']!='LoadGeometryRGBD':return original(pipe,scale)
        old=copy.deepcopy(pipe);reverse={'LoadGeometryRGBD':'LoadCourseRGBDImage','GeometryResize':'RGBDResize','PackGeometryRGBD':'PackRGBDDetInputs'}
        for row in old:row['type']=reverse.get(row['type'],row['type']);row.pop('geometry_index',None)
        return adapt_pipeline(original(old,scale),str(OUT/'geometry_index.json'))
    probe.scaled_pipeline=scaled;lr.evaluate()

def finish():
    w.finish();c=read(OUT/'comparison.json');c.update(primary='LR3e-5 single1536 45.03649817971514',treatment='seven_geometry_channels_four_scale_spatial_channel_gating',system_reference_APs=47.70413206494912,system_pending=True,submission_runs=0);save(OUT/'comparison.json',c)

w.build=build;w.compare_recipe=compare_recipe;w.register_hook=register_hook
if __name__=='__main__':
    capacity();mode=sys.argv[1]
    if mode=='cpu':cpu()
    elif mode in ('smoke','formal'):patch_initial();lr.train(mode=='smoke')
    elif mode=='evaluate':evaluate()
    elif mode=='finish':finish()
    else:raise ValueError(mode)
