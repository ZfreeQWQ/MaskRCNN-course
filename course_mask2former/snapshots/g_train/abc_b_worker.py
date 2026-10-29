"""B: only ordinary Hungarian shared-coordinate tiny quota changes."""
import copy,json,os,sys,time,hashlib,random
from pathlib import Path
import abc_b_spec as spec
import m2f_pointnoise_spec as parent_spec
for k,v in vars(spec).items():
    if k.isupper() or k in ('arm_spec','job','lease_end'):setattr(parent_spec,k,v)
# The accepted exact cache is bound before any phase starts.
ROOT=Path('/workspace');OUT=ROOT/'outputs'/spec.NAME
plan=json.loads((OUT/'plan.json').read_text());spec.EXACT_CACHE_SHA=parent_spec.EXACT_CACHE_SHA=plan['exact_cache_sha256']
import m2f_pointnoise_worker as parent
w=parent.w;read=parent.read;save=parent.save;sha=parent.sha

def build(args,cats):
    cfg=parent.build(args,cats);cfg.custom_imports.imports.append('rgbd.course_tiny_matching')
    cfg.model.type='TinyMatchedRGBDMask2Former';cfg.model.tiny_matching=dict(enabled=True,quota=1024,area_threshold=64)
    cfg.custom_hooks.append(dict(type='TinyMatchingAuditHook'));return cfg

def compare_recipe(cfg):
    from mmengine.config import Config
    assert sha(ROOT/spec.CP_RUN/'report.json')==spec.CP_REPORT
    old=Config.fromfile(str(ROOT/spec.CP_RUN/'config.py')).to_dict();new=cfg.to_dict()
    assert old['model']['type']=='NoisyPilotedRGBDMask2Former' and old['model']['course_pilot']==spec.OPTIONS
    old['model']['type']='TinyMatchedRGBDMask2Former';old['model']['tiny_matching']=dict(enabled=True,quota=1024,area_threshold=64)
    keys=('model','train_cfg','optim_wrapper','param_scheduler','randomness','train_dataloader','val_dataloader','val_evaluator','auto_scale_lr','env_cfg')
    changed=[k for k in keys if old[k]!=new[k]];assert not changed,changed
    return dict(status='passed',pilot_only=True,tiny_matching_only=True,baseline_report_sha256=spec.CP_REPORT,compared_keys=keys)

def space_gate(model,cfg):
    import numpy as np,torch
    from mmcv.ops import point_sample
    from mmengine.dataset import pseudo_collate
    from mmdet.registry import DATASETS
    from rgbd.course_tiny_matching import tiny_pixels,replace_tail
    dataset=DATASETS.build(cfg.train_dataloader.dataset);assert len(dataset)==14432
    order=list(range(len(dataset)));random.Random(20261013).shuffle(order);indices=order[:256]
    save(OUT/'space_manifest.json',dict(indices=indices,image_ids=[dataset.get_data_info(i)['img_id'] for i in indices],seed=20261013,selection_uses_GT=False,augmentation_seed_rule='20261013+dataset_index',quota=1024,points=12544,area=[1,64],gate_presence='point sampled target > 0',one_point_set_per_image=True))
    records=[];count=oldmiss=newmiss=0;rejected=0;cp_calls=0
    for n,index in enumerate(indices):
        w.deadline(spec.CPU_END);seed=20261013+index;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        sample=dataset.prepare_data(index)
        if sample is None:
            rejected+=1;records.append(dict(index=index,image_id=dataset.get_data_info(index)['img_id'],pipeline_rejected=True));continue
        processed=model.data_preprocessor(pseudo_collate([sample]),training=True)
        ds=processed['data_samples'][0];gt=model.panoptic_head.preprocess_gt([ds.gt_instances],[ds.gt_sem_seg if 'gt_sem_seg' in ds else None])[0]
        masks=gt.masks;pixels,shape=tiny_pixels(masks)
        # Independent audit draw has the same uniform law as ordinary matching.
        generator=torch.Generator().manual_seed(99173+index);coords=torch.rand((1,12544,2),generator=generator)
        global_before=torch.get_rng_state().clone();replacement,q=replace_tail(coords,pixels,shape,torch.Generator().manual_seed(731527+index))
        assert torch.equal(global_before,torch.get_rng_state()) and torch.equal(coords[:,:-1024],replacement[:,:-1024])
        ids=[i for i,p in pixels];om=[];nm=[]
        if ids:
            tiny=masks[ids,None].float();a=point_sample(tiny,coords.repeat(len(ids),1,1)).squeeze(1);b=point_sample(tiny,replacement.repeat(len(ids),1,1)).squeeze(1)
            om=[ids[i] for i,v in enumerate(a.max(1).values) if v<=0];nm=[ids[i] for i,v in enumerate(b.max(1).values) if v<=0]
        count+=len(ids);oldmiss+=len(om);newmiss+=len(nm)
        rec=dict(index=index,image_id=int(ds.img_id),tiny=len(ids),old_miss=om,new_miss=nm,replaced=q,mask_shape=list(masks.shape),mask_sha256=hashlib.sha256(masks.cpu().contiguous().numpy().tobytes()).hexdigest(),coords_sha256=hashlib.sha256(coords.numpy().tobytes()).hexdigest(),new_coords_sha256=hashlib.sha256(replacement.numpy().tobytes()).hexdigest())
        records.append(rec)
        if n%16==0:w.mark('CPU_space_gate',images=n+1,expected=256,tiny=count)
    passed=count>0 and oldmiss/count>=.10 and newmiss/count<=.01
    report=dict(status='passed' if passed else 'rejected',gate_passed=passed,images=256,processed=256-rejected,pipeline_rejected=rejected,tiny=count,old_missed=oldmiss,new_missed=newmiss,old_fraction=oldmiss/count if count else None,new_fraction=newmiss/count if count else None,old_min=.10,new_max=.01,records=records,CPU_only=True,model_forward_calls=0)
    save(OUT/'space_gate.json',report);return passed

def cpu():
    import torch,unittest
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmengine.utils import import_modules_from_strings
    import train_course_directions as t
    from test_abc_b import actual_matching_check
    w.deadline(spec.CPU_END);assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not Path('/dev/nvidiactl').exists()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(['test_abc_b','test_m2f_pointnoise','test_m2f_final','test_m2f_channel','test_course_directions_training','test_course_copy_paste','test_course_feature_models','test_course_resolution']))
    assert result.wasSuccessful() and not result.skipped
    register_all_modules(init_default_scope=True);args=w.args_for();cats=[dict(id=1,name='component',supercategory='component')]
    cfg=build(args,cats);checks=compare_recipe(cfg);import_modules_from_strings(**cfg.custom_imports)
    assert sha(args.checkpoint)==t.COCO_SHA
    state=t.base.component_state(torch.load(args.checkpoint,map_location='cpu')['state_dict'])
    torch.manual_seed(args.seed);old=MODELS.build(parent.build(args,cats).model);old.init_weights();t.initial_state(old,state);rng=torch.get_rng_state().clone()
    torch.manual_seed(args.seed);new=MODELS.build(cfg.model);new.init_weights();t.initial_state(new,state)
    assert torch.equal(rng,torch.get_rng_state()) and set(old.state_dict())==set(new.state_dict())
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    assert new.active_parameter_count()==spec.ACTIVE and sum(v.numel()*v.element_size() for v in new.state_dict().values())==spec.STATE
    actual_matching_check(new.panoptic_head,OUT/'matching_cpu_correctness.json')
    cfg.dump(str(OUT/'formal_template.py'));save(OUT/'cpu_model_binding.json',dict(status='passed',common_tensors_equal=True,RNG_equal=True,added_parameters=0,active=spec.ACTIVE,state_bytes=spec.STATE,COCO_sha256=t.COCO_SHA,recipe=checks))
    del old,state
    gate=space_gate(new,cfg);del new
    ref=ROOT/spec.REFERENCE;assert sha(ref/'report.json')==spec.REFERENCE_SHA and sha(ROOT/spec.CP_RUN/'iter_30000.pth')==spec.CP_WEIGHT
    assert sha(ref/'predictions.segm.json')==spec.EXACT_CACHE_SHA
    exact=read(ROOT/spec.EXACT);assert exact['status']=='passed' and exact['AP_runs']==1 and sha(ROOT/spec.EXACT)==spec.EXACT_SHA
    save(OUT/'reference_exact.json',exact);save(OUT/'reference_reuse.json',dict(path=spec.EXACT,sha256=sha(ROOT/spec.EXACT),new_baseline_AP_runs=0))
    w.deadline(spec.CPU_END);save(OUT/'cpu_acceptance.json',dict(status='passed',tests=result.testsRun,zero_skip=True,space_gate_passed=gate,binding_sha256=sha(OUT/'cpu_model_binding.json'),reference_sha256=sha(OUT/'reference_exact.json'),unix=time.time()))

def register_hook():
    parent.register_hook()
    from mmengine.hooks import Hook
    from mmengine.registry import HOOKS
    @HOOKS.register_module()
    class TinyMatchingAuditHook(Hook):
        priority='VERY_LOW'
        def before_train(self,runner):assert runner.model.panoptic_head.match_activity==dict(calls=0,tiny_targets=0,replaced_points=0)
        def before_val_epoch(self,runner):self.snapshot=copy.deepcopy(runner.model.panoptic_head.match_activity)
        def after_val_epoch(self,runner,metrics=None):assert self.snapshot==runner.model.panoptic_head.match_activity
        def after_train(self,runner):
            a=runner.model.panoptic_head.match_activity;layers=len(runner.model.panoptic_head.transformer_decoder.layers)+1
            assert a['calls']==runner.max_iters*2*layers and a['replaced_points']>0 and a['tiny_targets']>0
            save(Path(runner.work_dir)/'matching_audit.json',dict(status='passed',activity=a,iterations=runner.max_iters,quota=1024,shared_points=12544,post_matching_loss_unchanged=True,inference_matching_calls=0))

def finish():
    w.finish();c=read(OUT/'comparison.json');c.update(primary='pointnoise_single1536',treatment='tiny_shared_matching_points_only',A_not_stacked=True);save(OUT/'comparison.json',c)
w.build=build;w.compare_recipe=compare_recipe;w.register_hook=register_hook
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='cpu':cpu()
    elif mode in ('smoke','formal'):w.args_for_saved=w.args_for(mode=='smoke');w.train(mode=='smoke')
    elif mode=='evaluate':w.evaluate()
    elif mode=='finish':finish()
    else:raise ValueError(mode)
