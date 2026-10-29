"""Single-point-noise adapters; reuse accepted training/evaluation implementation."""
import copy,json,os,sys,time
from pathlib import Path
import m2f_pointnoise_spec as spec
import m2f_final_spec as final
final.arm_spec=spec.arm_spec
final.EXACT=spec.EXACT;final.EXACT_CACHE_SHA=spec.EXACT_CACHE_SHA
os.environ['FINAL_ARM']='pilot'
import m2f_final_worker as w
from launch_nyu_sweep import atomic_json as save,sha256 as sha
ROOT=w.ROOT;OUT=w.OUT;s=w.s
read=w.read

def build(args,categories):
    import train_course_directions as t
    cfg=t.build_config(args,categories)
    assert args.course_copy_paste and args.multiscale_depth and args.reliability_gate
    assert not any(getattr(args,k) for k in ('course_detail','course_depth_reliability','course_high_resolution'))
    cfg.custom_imports.imports.append('rgbd.course_pointnoise')
    cfg.model.type='NoisyPilotedRGBDMask2Former';cfg.model.course_pilot=copy.deepcopy(spec.OPTIONS)
    cfg.custom_hooks.extend([dict(type='ChannelAuditHook'),dict(type='PointNoiseAuditHook')])
    return cfg

def compare_recipe(cfg):
    from mmengine.config import Config
    assert sha(ROOT/s.CP_RUN/'report.json')==s.CP_REPORT
    old=Config.fromfile(str(ROOT/s.CP_RUN/'config.py')).to_dict();new=cfg.to_dict()
    assert old['model']['type']=='PilotedRGBDMask2Former'
    assert old['model']['course_pilot']==dict(enabled=True,max_queries=16,weight=.5,no_noise=True)
    old['model']['type']='NoisyPilotedRGBDMask2Former';old['model']['course_pilot']=copy.deepcopy(spec.OPTIONS)
    keys=('model','train_cfg','optim_wrapper','param_scheduler','randomness','train_dataloader',
        'val_dataloader','val_evaluator','auto_scale_lr','env_cfg')
    changed=[k for k in keys if old[k]!=new[k]];assert not changed,changed
    return dict(status='passed',pilot_only=True,point_noise_only=True,baseline_report_sha256=s.CP_REPORT,compared_keys=keys)

def cpu():
    import unittest,torch
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    import train_course_directions as t
    from test_m2f_pointnoise import actual_noise_check
    w.deadline(s.CPU_END)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not Path('/dev/nvidiactl').exists()
    tests=unittest.defaultTestLoader.loadTestsFromNames(['test_m2f_pointnoise','test_m2f_final','test_m2f_channel',
        'test_course_directions_training','test_course_copy_paste','test_course_feature_models','test_course_resolution'])
    result=unittest.TextTestRunner(verbosity=2).run(tests);assert result.wasSuccessful() and not result.skipped
    w.mark('cpu_initialization',tests=result.testsRun)
    register_all_modules(init_default_scope=True)
    args=w.args_for();cats=[dict(id=1,name='component',supercategory='component')]
    cfg=build(args,cats);checks=compare_recipe(cfg);import_modules_from_strings(**cfg.custom_imports)
    assert sha(args.checkpoint)==t.COCO_SHA
    state=t.base.component_state(torch.load(args.checkpoint,map_location='cpu')['state_dict'])
    torch.manual_seed(args.seed);old=MODELS.build(t.build_config(args,cats).model);old.init_weights()
    t.initial_state(old,state);rng=torch.get_rng_state().clone()
    torch.manual_seed(args.seed);new=MODELS.build(cfg.model);new.init_weights();t.initial_state(new,state)
    assert torch.equal(rng,torch.get_rng_state())
    a=old.state_dict();b=new.state_dict();assert set(a)==set(b) and all(torch.equal(v,b[k]) for k,v in a.items())
    assert new.active_parameter_count()==s.ACTIVE and sum(v.numel()*v.element_size() for v in b.values())==s.STATE
    actual_noise_check(new.panoptic_head,OUT/'noise_cpu_correctness.json')
    cfg.dump(str(OUT/'formal_template.py'))
    save(OUT/'cpu_model_binding.json',dict(status='passed',common_tensors_equal=True,RNG_equal=True,added_parameters=0,
        active=s.ACTIVE,state_bytes=s.STATE,COCO_sha256=t.COCO_SHA,recipe=checks))
    del old,new,state,a,b
    ref=ROOT/s.REFERENCE;r=read(ref/'report.json')
    assert sha(ref/'report.json')==s.REFERENCE_SHA and sha(ROOT/s.CP_RUN/'iter_30000.pth')==s.CP_WEIGHT
    assert sha(ref/'predictions.segm.json')==spec.EXACT_CACHE_SHA
    assert sha(ref/'input_hashes.json')==r['input_hashes_sha256']
    exact=read(ROOT/spec.EXACT);assert exact['status']=='passed' and exact['AP_runs']==1
    assert read(ROOT/spec.PARENT/'arms/pilot/comparison.json')['metrics']==exact['metrics']
    rounded={k:r['metrics']['course/segm_mAP'+suffix] for k,suffix in zip(('AP','AP50','AP75','APs','APm','APl'),('','_50','_75','_s','_m','_l'))}
    assert all(abs(exact['metrics'][k]-v)<=.00051 for k,v in rounded.items())
    save(OUT/'reference_exact.json',dict(exact,cache_sha256=spec.EXACT_CACHE_SHA,reference_report_sha256=s.REFERENCE_SHA))
    save(OUT/'reference_reuse.json',dict(path=spec.EXACT,sha256=sha(ROOT/spec.EXACT),new_baseline_AP_runs=0))
    save(OUT/'auxiliary_CP_exact.json',read(ROOT/spec.AUX_EXACT))
    w.deadline(s.CPU_END)
    save(OUT/'cpu_acceptance.json',dict(status='passed',tests=result.testsRun,zero_skip=True,
        binding_sha256=sha(OUT/'cpu_model_binding.json'),noise_correctness_sha256=sha(OUT/'noise_cpu_correctness.json'),
        reference_sha256=sha(OUT/'reference_exact.json'),unix=time.time()))

old_register=w.register_hook
def register_hook():
    old_register()
    from mmengine.hooks import Hook
    from mmengine.registry import HOOKS
    @HOOKS.register_module()
    class PointNoiseAuditHook(Hook):
        priority='VERY_LOW'
        def before_train(self,runner):
            assert all(v==0 for v in runner.model.panoptic_head.noise_activity.values())
        def before_val_epoch(self,runner):self.snapshot=copy.deepcopy(runner.model.panoptic_head.noise_activity)
        def after_val_epoch(self,runner,metrics=None):
            assert self.snapshot==runner.model.panoptic_head.noise_activity,'Inference used noise/GT auxiliary'
        def after_train(self,runner):
            h=runner.model.panoptic_head;a=copy.deepcopy(h.noise_activity)
            assert a['train_calls']==runner.max_iters and a['flips']>0 and a['noised_guides']>0
            assert a['guides']==h.pilot_queries*len(h.transformer_decoder.layers)
            assert a['flips']<=.2*a['guide_foreground']
            save(Path(runner.work_dir)/'noise_audit.json',dict(status='passed',activity=a,
                point_noise_ratio=.2,clean_loss_targets=True,inference_noise_calls=0,added_parameters=0))

def finish():
    w.finish()
    c=read(OUT/'comparison.json');aux=read(OUT/'auxiliary_CP_exact.json')['metrics']
    c.update(primary='CP + clean mask-pilot, same 1536',treatment='point-noise-only',auxiliary_CP=aux,
        delta_vs_CP_points={k:100*(c['metrics'][k]-aux[k]) for k in aux},
        complex_best_APs=41.76790163964242,does_not_stack_with_complex_best=True)
    save(OUT/'comparison.json',c)

w.build=build;w.compare_recipe=compare_recipe;w.register_hook=register_hook
if __name__=='__main__':
    mode=sys.argv[1]
    if mode in ('smoke','formal'):
        w.args_for_saved=w.args_for(mode=='smoke');w.train(mode=='smoke')
    elif mode=='evaluate':w.evaluate()
    elif mode=='cpu':cpu()
    elif mode=='finish':finish()
    else:raise ValueError(mode)
