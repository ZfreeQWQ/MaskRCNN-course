"""LR-only wrapper; inherited B numerical sources remain byte-identical."""
import copy,json,os,sys,time
from pathlib import Path
import lr_spec as spec
import abc_b_spec
for k,v in vars(spec).items():
    if k.isupper() or k in ('arm_spec','job','lease_end'):setattr(abc_b_spec,k,v)
import abc_b_worker as b
from lr_contract import check_delta
w=b.w;ROOT=b.ROOT;OUT=b.OUT;read=b.read;save=b.save;sha=b.sha
original_build=b.build
original_args=w.args_for
def args_for(smoke=False):
    args=original_args(smoke);args.lr=spec.LR;return args
def build(args,cats):
    # Original builder's old 1e-5 guard remains intact; only its resulting LR changes.
    assert args.lr in (spec.BASE_LR,spec.LR)
    old=copy.copy(args);old.lr=spec.BASE_LR
    cfg=original_build(old,cats);cfg.optim_wrapper.optimizer.lr=spec.LR
    cfg.custom_hooks.append(dict(type='LearningRateAuditHook'))
    return cfg
def compare_recipe(cfg):
    from mmengine.config import Config
    assert sha(ROOT/spec.CP_RUN/'report.json')==spec.CP_REPORT
    checks=check_delta(Config.fromfile(str(ROOT/spec.CP_RUN/'config.py')).to_dict(),cfg.to_dict())
    checks.update(pilot_only=True,baseline_report_sha256=spec.CP_REPORT)
    return checks
def groups(model,optim):
    by_id={id(p):name for name,p in model.named_parameters()};result={}
    for group in optim.optimizer.param_groups:
        for p in group['params']:
            result[by_id[id(p)]]=dict(lr=group['lr'],weight_decay=group['weight_decay'])
    return result
def cpu():
    import torch,unittest
    from mmengine.optim import build_optim_wrapper
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    import train_course_directions as t
    from test_abc_b import actual_matching_check
    w.deadline(spec.CPU_END);assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not Path('/dev/nvidiactl').exists()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(
        ['test_lr','test_abc_b','test_m2f_pointnoise','test_m2f_final','test_m2f_channel','test_course_directions_training','test_course_copy_paste','test_course_feature_models','test_course_resolution']))
    assert result.wasSuccessful() and not result.skipped
    register_all_modules(init_default_scope=True);args=args_for();cats=[dict(id=1,name='component',supercategory='component')]
    oldargs=copy.copy(args);oldargs.lr=spec.BASE_LR
    oldcfg=original_build(oldargs,cats);cfg=build(args,cats);checks=compare_recipe(cfg)
    import_modules_from_strings(**cfg.custom_imports)
    assert sha(args.checkpoint)==t.COCO_SHA
    state=t.base.component_state(torch.load(args.checkpoint,map_location='cpu')['state_dict'])
    torch.manual_seed(args.seed);old=MODELS.build(oldcfg.model);old.init_weights();t.initial_state(old,state);rng=torch.get_rng_state().clone()
    torch.manual_seed(args.seed);new=MODELS.build(cfg.model);new.init_weights();t.initial_state(new,state)
    assert torch.equal(rng,torch.get_rng_state()) and set(old.state_dict())==set(new.state_dict())
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    oldopt=build_optim_wrapper(old,oldcfg.optim_wrapper);newopt=build_optim_wrapper(new,cfg.optim_wrapper)
    a,c=groups(old,oldopt),groups(new,newopt);assert a.keys()==c.keys()
    assert all(abs(c[k]['lr']/a[k]['lr']-3)<1e-12 and c[k]['weight_decay']==a[k]['weight_decay'] for k in a)
    save(OUT/'optimizer_group_binding.json',dict(status='passed',old=a,new=c,all_groups_ratio=3,optimizer_updates=0))
    assert new.active_parameter_count()==spec.ACTIVE and sum(v.numel()*v.element_size() for v in new.state_dict().values())==spec.STATE
    actual_matching_check(new.panoptic_head,OUT/'matching_cpu_correctness.json')
    cfg.dump(str(OUT/'formal_template.py'))
    save(OUT/'cpu_model_binding.json',dict(status='passed',common_tensors_equal=True,RNG_equal=True,added_parameters=0,
         active=spec.ACTIVE,state_bytes=spec.STATE,COCO_sha256=t.COCO_SHA,recipe=checks,optimizer_groups_sha256=sha(OUT/'optimizer_group_binding.json')))
    del old,new,state,oldopt,newopt
    for name,digest in [('space_gate.json',spec.SPACE_SHA),('space_manifest.json',spec.SPACE_MANIFEST_SHA)]:
        source=ROOT/spec.PARENT/name;assert sha(source)==digest;save(OUT/name,read(source))
    assert read(OUT/'space_gate.json')['gate_passed']
    save(OUT/'space_gate_reuse.json',dict(status='passed',path=spec.PARENT,gate_sha256=spec.SPACE_SHA,
         rationale='LR does not enter augmentation or matching coordinates; model/data/source unchanged',new_space_evaluations=0))
    ref=ROOT/spec.REFERENCE
    assert sha(ref/'report.json')==spec.REFERENCE_SHA and sha(ROOT/spec.CP_RUN/'iter_30000.pth')==spec.CP_WEIGHT
    assert sha(ref/'predictions.segm.json')==spec.EXACT_CACHE_SHA and sha(ROOT/spec.EXACT)==spec.EXACT_SHA
    exact=read(ROOT/spec.EXACT);assert exact['status']=='passed' and exact['AP_runs']==1
    save(OUT/'reference_exact.json',exact);save(OUT/'reference_reuse.json',dict(path=spec.EXACT,sha256=spec.EXACT_SHA,new_baseline_AP_runs=0))
    w.deadline(spec.CPU_END);save(OUT/'cpu_acceptance.json',dict(status='passed',tests=result.testsRun,zero_skip=True,
        space_gate_passed=True,space_gate_reused=True,binding_sha256=sha(OUT/'cpu_model_binding.json'),unix=time.time()))
def register_hook():
    b.register_hook()
    from mmengine.hooks import Hook
    from mmengine.registry import HOOKS
    @HOOKS.register_module()
    class LearningRateAuditHook(Hook):
        def before_train(self,runner):
            self.initial=groups(runner.model,runner.optim_wrapper);self.observed=[]
            expected=read(OUT/'optimizer_group_binding.json')['new'];assert self.initial==expected
        def before_train_iter(self,runner,batch_idx,data_batch=None):
            milestones=list(runner.cfg.param_scheduler[0]['milestones'])
            if runner.iter not in [0,*milestones]:return
            current=groups(runner.model,runner.optim_wrapper);factor=.1**sum(runner.iter>=m for m in milestones)
            assert all(abs(current[k]['lr']/self.initial[k]['lr']-factor)<1e-12 for k in current)
            self.observed.append(dict(iteration=runner.iter,scale=factor,rates=sorted(set(x['lr'] for x in current.values()))))
        def after_train(self,runner):
            save(Path(runner.work_dir)/'lr_audit.json',dict(status='passed',base_lr=spec.LR,all_initial_groups=self.initial,
                 observed=self.observed,iterations=runner.max_iters,base_group_multiplier=3))
def train(smoke):
    import train_course_directions as t
    old_validate=t.validate_direction_args
    def validate(args):
        assert args.lr in (spec.BASE_LR,spec.LR)
        old=copy.copy(args);old.lr=spec.BASE_LR;old_validate(old)
    t.validate_direction_args=validate
    w.args_for_saved=args_for(smoke);assert w.args_for_saved.lr==spec.LR
    w.train(smoke)
def evaluate():
    import run_course_directions as q
    q.BASE_RECIPE=dict(q.BASE_RECIPE,lr=spec.LR)
    import course_directions_diagnostics as diag
    diag.BASE_RECIPE=dict(diag.BASE_RECIPE,lr=spec.LR)
    w.evaluate()
def finish():
    w.finish();c=read(OUT/'comparison.json')
    c.update(primary='B tiny-matching single1536',treatment='base_LR_1e-5_to_3e-5_only',
             complex_best_APs=42.46308704182373,automatic_combination=False,submission_runs=0)
    save(OUT/'comparison.json',c)
w.build=build;w.compare_recipe=compare_recipe;w.register_hook=register_hook;w.args_for=args_for
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='cpu':cpu()
    elif mode in ('smoke','formal'):train(mode=='smoke')
    elif mode=='evaluate':evaluate()
    elif mode=='finish':finish()
    else:raise ValueError(mode)
