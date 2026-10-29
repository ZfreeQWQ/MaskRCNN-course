"""Isolated adapters around the frozen CP trainer and resolution evaluator."""
import copy,json,math,os,sys,time
from pathlib import Path
import m2f_channel_spec as s
ROOT=Path('/workspace');OUT=ROOT/'outputs'/s.NAME
from launch_nyu_sweep import atomic_json as save,sha256 as sha

def read(p):return json.loads(Path(p).read_text())
def mark(phase,**kw):save(OUT/'progress.json',dict(phase=phase,unix=time.time(),**kw))
def deadline(end):
    if time.time()>=min(end,s.DEADLINE):raise TimeoutError('Immutable channel budget')

def build(args,categories):
    import train_course_directions as t
    cfg=t.build_config(args,categories)
    if not (args.course_copy_paste and args.multiscale_depth and args.reliability_gate) or any(
        getattr(args,k) for k in ('course_detail','course_depth_reliability','course_high_resolution')):
        raise ValueError('Only CP plus channel calibration')
    cfg.custom_imports.imports.append('rgbd.course_channel')
    cfg.model.type='ChannelRGBDMask2Former';cfg.model.course_channel=dict(enabled=True)
    cfg.custom_hooks.append(dict(type='ChannelAuditHook'))
    return cfg

def args_for(smoke=False):
    import train_course_directions as t
    import run_course_directions as q
    args=['train','--workspace','/workspace','--data-root','/workspace/'+q.DATA,
        '--checkpoint','/workspace/'+q.CHECKPOINT,'--output',str(OUT/('smoke' if smoke else 'formal')/'runs'/s.JOB),
        '--seed','20261013','--multiscale-depth','--reliability-gate','--course-copy-paste']
    for k,v in q.stage_recipe(smoke).items():args+=['--'+k.replace('_','-'),str(v)]
    if smoke:args+=['--functional-smoke']
    return t.parser().parse_args(args)

def compare_recipe(cfg):
    from mmengine.config import Config
    cp=ROOT/s.CP_RUN
    assert sha(cp/'report.json')==s.CP_REPORT
    old=Config.fromfile(str(cp/'config.py')).to_dict();new=cfg.to_dict()
    old['model']['type']='ChannelRGBDMask2Former';old['model']['course_channel']=dict(enabled=True)
    keys=('model','train_cfg','optim_wrapper','param_scheduler','randomness','train_dataloader',
        'val_dataloader','val_evaluator','auto_scale_lr','env_cfg')
    changed=[k for k in keys if old[k]!=new[k]]
    assert not changed,changed
    return dict(status='passed',channel_only=True,CP_report_sha256=s.CP_REPORT,compared_keys=keys)

def exact_eval(path,outfile):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import run_course_directions as q
    ann=ROOT/q.DATA/'annotations/instances_val.json';assert sha(ann)==q.VAL_SHA
    rows=read(path);coco=COCO(str(ann));ids=[x['id'] for x in coco.dataset['images']]
    allowed=set(ids)
    assert len(ids)==2584 and all(x['image_id'] in allowed for x in rows)
    # segm area must come from masks, not bbox (COCO small-area convention).
    for row in rows:row.pop('bbox',None)
    ev=COCOeval(coco,coco.loadRes(rows),'segm');ev.params.imgIds=ids;ev.params.maxDets=[1,10,100]
    ev.evaluate();ev.accumulate();ev.summarize()
    values={k:float(v) for k,v in zip(('AP','AP50','AP75','APs','APm','APl'),ev.stats[:6])}
    assert all(math.isfinite(v) and 0<=v<=1 for v in values.values())
    save(outfile,dict(status='passed',metrics=values,cache_sha256=sha(path),val_sha256=sha(ann),images=2584,AP_runs=1,unix=time.time()))

def cpu():
    deadline(s.CPU_END)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not Path('/dev/nvidiactl').exists()
    import unittest,torch
    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    import train_course_directions as t
    tests=unittest.defaultTestLoader.loadTestsFromNames(['test_m2f_channel','test_course_directions_training',
        'test_course_copy_paste','test_course_feature_models','test_course_resolution'])
    result=unittest.TextTestRunner(verbosity=2).run(tests)
    assert result.wasSuccessful() and not result.skipped
    mark('cpu_initialization',tests=result.testsRun)
    register_all_modules(init_default_scope=True)
    args=args_for();cats=[dict(id=1,name='component',supercategory='component')]
    cfg=build(args,cats);checks=compare_recipe(cfg);import_modules_from_strings(**cfg.custom_imports)
    assert sha(args.checkpoint)==t.COCO_SHA
    state=t.base.component_state(torch.load(args.checkpoint,map_location='cpu')['state_dict'])
    torch.manual_seed(args.seed);old=MODELS.build(t.build_config(args,cats).model);old.init_weights()
    t.initial_state(old,state);oldrng=torch.get_rng_state().clone()
    torch.manual_seed(args.seed);new=MODELS.build(cfg.model);new.init_weights();t.initial_state(new,state)
    assert torch.equal(oldrng,torch.get_rng_state())
    a=old.state_dict();b=new.state_dict();assert all(torch.equal(v,b[k]) for k,v in a.items())
    added=set(b)-set(a);assert added and all(k.startswith('spatial_fusion.channel_mlps.') for k in added)
    assert new.active_parameter_count()==s.ACTIVE and sum(x.numel()*x.element_size() for x in b.values())==s.STATE
    assert sum(b[k].numel() for k in added)==s.ADDED
    assert all(not bool(mlp[-1].weight.count_nonzero()) and not bool(mlp[-1].bias.count_nonzero()) for mlp in new.spatial_fusion.channel_mlps)
    cfg.dump(str(OUT/'formal_template.py'))
    save(OUT/'cpu_model_binding.json',dict(status='passed',common_tensors_equal=True,RNG_equal=True,
        added_parameters=s.ADDED,active=s.ACTIVE,state_bytes=s.STATE,COCO_sha256=t.COCO_SHA,recipe=checks))
    del new,old,state,a,b
    ref=ROOT/s.REFERENCE;assert sha(ref/'report.json')==s.REFERENCE_SHA
    assert sha(ROOT/s.CP_RUN/'iter_30000.pth')==s.CP_WEIGHT
    cache=ref/'predictions.segm.json';assert cache.is_file()
    oldreport=read(ref/'report.json');assert sha(ref/'input_hashes.json')==oldreport['input_hashes_sha256']
    mark('cpu_reference_AP');exact_eval(cache,OUT/'reference_exact.json')
    rounded={k:oldreport['metrics']['course/segm_mAP'+suffix] for k,suffix in zip(('AP','AP50','AP75','APs','APm','APl'),('','_50','_75','_s','_m','_l'))}
    assert all(abs(read(OUT/'reference_exact.json')['metrics'][k]-v)<=.00051 for k,v in rounded.items())
    deadline(s.CPU_END)
    save(OUT/'cpu_acceptance.json',dict(status='passed',tests=result.testsRun,zero_skip=True,
        binding_sha256=sha(OUT/'cpu_model_binding.json'),reference_sha256=sha(OUT/'reference_exact.json'),unix=time.time()))

def register_hook():
    import torch
    from mmengine.hooks import Hook
    from mmengine.registry import HOOKS
    import train_course_directions as t
    @HOOKS.register_module()
    class ChannelAuditHook(Hook):
        priority='VERY_LOW'
        def before_train(self,runner):
            self.initial={k:v.detach().cpu().clone() for k,v in runner.model.state_dict().items() if k.startswith('spatial_fusion.channel_mlps.')}
            assert len(self.initial)==16
            self.steps=0;self.wall=[];self.last=None
        def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
            self.steps+=1;now=time.time()
            if self.last is not None and self.steps<=200:self.wall.append(now-self.last)
            self.last=now
            for k,p in runner.model.named_parameters():
                if k in self.initial and p.grad is not None and not bool(torch.isfinite(p.grad).all()):raise RuntimeError('Nonfinite channel gradient')
            if self.steps%10==0:mark('training',step=self.steps,max_steps=runner.max_iters)
            if self.steps==200 and runner.max_iters==30000:
                import statistics
                mean=statistics.mean(self.wall[-150:])
                # All six future validations, checkpoint/exit margin; once only.
                estimate=(30000-self.steps)*mean*1.15+6*1500+300
                limit=min(read(OUT/'phase_deadlines.json')['worker_end'],s.FORMAL_END)
                gate=dict(status='passed' if now+estimate<limit else 'failed',step=200,
                    steady_mean_seconds=mean,remaining_estimate=estimate,remaining_seconds=limit-now,safety=1.15)
                save(OUT/'formal_throughput_gate.json',gate);assert gate['status']=='passed',gate
        def before_val_epoch(self,runner):self.ids=[]
        def after_val_iter(self,runner,batch_idx,data_batch=None,outputs=None):
            self.ids.extend(int(x.img_id) for x in outputs)
            if len(self.ids)%50==0:mark('validation',step=runner.iter,images=len(self.ids))
        def after_val_epoch(self,runner,metrics=None):
            expected=[x['id'] for x in read(runner.cfg.val_evaluator.ann_file)['images']]
            assert self.ids==expected
            save(Path(runner.work_dir)/f'validation_ids_{runner.iter}.json',self.ids)
        def after_train(self,runner):
            weights=runner.model.state_dict();changed={k:not torch.equal(v,weights[k].detach().cpu()) for k,v in self.initial.items()}
            assert all(changed.values()),changed
            assert all(t.finite_tree(weights[k]) for k in changed)
            calls=runner.model.spatial_fusion.channel_calls
            assert calls['train']==runner.max_iters and calls['train_samples']==runner.max_iters*2
            save(Path(runner.work_dir)/'channel_audit.json',dict(status='passed',updated=changed,iterations=self.steps,calls=calls))
    return ChannelAuditHook

def train(smoke):
    import train_course_directions as t
    from mmengine.config import Config
    register_hook();original=t.build_config
    # build() calls the original; avoid recursion without altering frozen source.
    def calibrated(args,cats):
        t.build_config=original
        try:return build(args,cats)
        finally:t.build_config=calibrated
    def verify(args,cats,cfg):
        formal=copy.copy(args);formal.functional_smoke=False
        formal.iters,formal.val_interval,formal.keep_checkpoints=30000,5000,6
        expected=calibrated(formal,cats);checks=compare_recipe(expected)
        template=Config.fromfile(str(OUT/'formal_template.py'))
        assert expected.model==template.model and expected.optim_wrapper==template.optim_wrapper
        checks['cpu_actual_binding']=sha(OUT/'cpu_model_binding.json');return checks
    t.build_config=calibrated;t.verify_reference_recipe=verify
    t.parser=lambda: type('Parser',(),{'parse_args':lambda self:args_for_saved})()
    # Parse before replacing parser.
    t.main()

def evaluate():
    import course_resolution_probe as probe
    import course_directions_diagnostics as diag
    from types import SimpleNamespace
    from mmdet.evaluation.metrics.coco_metric import CocoMetric
    from pycocotools.cocoeval import COCOeval
    run=OUT/'formal/runs'/s.JOB
    diag.jobs=lambda:[s.job()]
    r,ck,source,_=diag.verify_direction(run)
    probe.verify_run=lambda p:diag.verify_direction(Path(p))[:3]
    probe.CHECKPOINT_SHA=r['checkpoint_sha256']
    # Capture the exact result from the evaluator's one existing COCO pass.
    original=COCOeval.summarize
    def summarize(ev):
        result=original(ev)
        if ev.params.iouType=='segm':
            save(OUT/'new_exact.json',dict(status='passed',metrics={k:float(v) for k,v in zip(('AP','AP50','AP75','APs','APm','APl'),ev.stats[:6])},AP_runs=1,unix=time.time()))
        return result
    COCOeval.summarize=summarize
    probe.run(SimpleNamespace(run=str(run),output=str(OUT/'evaluate_1536'),mode='evaluate',course_eval_scale=1536))
    COCOeval.summarize=original
    full_timing(run)

def full_timing(run):
    import torch,statistics,numpy as np
    from mmengine.config import Config
    from mmengine.dataset import Compose,pseudo_collate
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from course_demo import encode_predictions
    from course_resolution_probe import scaled_pipeline,safe_request,FIXED_IDS
    allresults={}
    for arm,path in [('baseline',ROOT/s.CP_RUN),('channel',run)]:
        deadline(s.EVAL_END);cfg=Config.fromfile(str(path/'config.py'));import_modules_from_strings(**cfg.custom_imports)
        saved=torch.load(path/'iter_30000.pth',map_location='cpu');model=MODELS.build(cfg.model)
        model.load_state_dict(saved['state_dict'],strict=True);del saved;model.cuda().eval()
        doc=read(cfg.val_evaluator.ann_file);image=next(x for x in doc['images'] if x['id']==FIXED_IDS[0])
        req=safe_request(cfg.val_dataloader.dataset.data_root,image);compose=Compose(scaled_pipeline(cfg.val_dataloader.dataset.pipeline,1536));times=[];peaks=[]
        with torch.inference_mode():
            for i in range(25):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
                sample=compose(copy.deepcopy(req));ds=sample['data_samples']
                if 'gt_instances' in ds:
                    assert len(ds.gt_instances)==0;del ds.gt_instances
                pred=model.test_step(pseudo_collate([sample]))[0].cpu().pred_instances
                rows=encode_predictions(pred.masks.numpy(),pred.scores.numpy(),image['id']);json.dumps(rows,allow_nan=False)
                torch.cuda.synchronize();ms=(time.perf_counter()-begin)*1000
                if i>=5:times.append(ms);peaks.append(torch.cuda.max_memory_allocated()/2**20)
                del pred,sample,rows;mark('paired_full_timing',arm=arm,iteration=i+1)
        allresults[arm]=dict(warmup=5,repeats=20,milliseconds=times,mean_ms=statistics.mean(times),
            p95_ms=float(np.percentile(times,95)),peak_allocated_mib=max(peaks),image_id=FIXED_IDS[0],
            scope='RGB-D IO/resize/pack/H2D/forward/D2H/RLE/JSON; synchronized FP32 B1; no model construction')
        del model;torch.cuda.empty_cache()
    save(OUT/'paired_timing.json',allresults)

def finish():
    from pycocotools.coco import COCO
    from pycocotools import mask as mu
    import run_course_directions as q
    deadline(s.DEADLINE);assert not Path('/dev/nvidiactl').exists()
    old=read(OUT/'reference_exact.json');new=read(OUT/'new_exact.json');r=read(OUT/'evaluate_1536/report.json')
    assert read(OUT/'evaluate_1536/input_hashes.json')==read(ROOT/s.REFERENCE/'input_hashes.json')
    assert r['active_parameters']==s.ACTIVE and r['state_tensor_bytes']==s.STATE and r['completed_images']==2584
    # Post-hoc matching only, no second COCO AP. Fixed eight cases; full caches retained.
    coco=COCO(str(ROOT/q.DATA/'annotations/instances_val.json'));cases={}
    for i in q.FIXED_IDS:
        anns=[a for a in coco.loadAnns(coco.getAnnIds(imgIds=[i])) if not a.get('iscrowd',0)]
        gt=[coco.annToRLE(a) for a in anns];entry={}
        for label,root in [('baseline',ROOT/s.REFERENCE),('channel',OUT/'evaluate_1536')]:
            rows=read(root/f'case_{i}_predictions.json');ious=mu.iou([x['segmentation'] for x in rows],gt,[0]*len(gt));summary={}
            for threshold in (.5,.75):
                matched=set();fps=0
                for n in sorted(range(len(rows)),key=lambda k:-rows[k]['score']):
                    options=[j for j in range(len(anns)) if j not in matched and ious[n,j]>=threshold]
                    if options:matched.add(max(options,key=lambda j:ious[n,j]))
                    else:fps+=1
                summary[str(threshold)]=dict(matched_ids=[anns[j]['id'] for j in sorted(matched)],FP=fps)
            entry[label]=summary
        entry['GT_areas']={a['id']:a['area'] for a in anns};cases[i]=entry
    save(OUT/'paired_cases.json',cases)
    save(OUT/'comparison.json',dict(status='passed',baseline=old['metrics'],metrics=new['metrics'],
        delta_points={k:100*(new['metrics'][k]-old['metrics'][k]) for k in new['metrics']},
        small_positive_retained=new['metrics']['APs']>old['metrics']['APs'],single_seed=True,official_complex_best_unchanged=True,
        paired_timing=read(OUT/'paired_timing.json'),new_AP_runs=1,baseline_cached_AP_runs=1))

if __name__=='__main__':
    mode=sys.argv[1]
    if mode in ('smoke','formal'):
        args_for_saved=args_for(mode=='smoke');train(mode=='smoke')
    else:globals()[mode]()
