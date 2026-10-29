"""One source-specific verifier pilot; immutable 120min, no full val."""
import copy, hashlib, math, os, statistics, time, unittest
from pathlib import Path
import run_m2f_fine512 as old
import run_m2f_tiny_positive as w
START=1792528506
PREP_END,CAPTURE_END,LABEL_END,TRAIN_END,ANALYSIS_END,DEADLINE=(START+n for n in (1200,3300,4500,5700,6600,7200))
NAME='course_m2f_fine512_adapt_20261021_v1'
OWNER='72d613e71efb4e73ad0506c3133ea8eb0'
ROOT=Path('/workspace');OUT=ROOT/'outputs'/NAME
PRIOR=ROOT/'outputs/course_m2f_fine512_20261021_v1'
AUDIT=ROOT/'outputs/course_m2f_tiny_rankaudit_20261021_v1'
read,save,sha=w.read,w.save,w.sha
old.START,old.DEADLINE,old.OUT=START,DEADLINE,OUT

def mark(phase,**kw):save(OUT/'progress.json',dict(phase=phase,unix=time.time(),**kw))
def cutoff(end):old.cutoff(end)

def space_gate(records):
    pos=[r for r in records if r['target']==1];neg=[r for r in records if r['target']==0]
    counts=dict(positive_gt=len({r['gt_id'] for r in pos}),positive_scenes=len({r['scene'] for r in pos}),
        positive_area64_gt=len({r['gt_id'] for r in pos if r['gt_area']<=64}),negative_candidates=len(neg),
        negative_scenes=len({r['scene'] for r in neg}))
    passed=counts['positive_gt']>=32 and counts['positive_scenes']>=16 and counts['positive_area64_gt']>=8 and len(neg)>=128 and counts['negative_scenes']>=32
    return passed,counts

class Contracts(unittest.TestCase):
    def test_budget(self):self.assertEqual(DEADLINE-START,7200);self.assertEqual(PREP_END-START,1200)
    def test_gate_unique_not_candidates(self):
        rows=[dict(target=1,gt_id=1,scene='a',gt_area=10)]*1000
        self.assertFalse(space_gate(rows)[0])
    def test_source_prefix_unchanged(self):
        import m2f_tiny_verifier as tiny
        from test_m2f_tiny_verifier import row
        cur=[row()];a=[row(300)];b=[row(500)]
        self.assertEqual(tiny.eligible(cur,a),tiny.eligible(cur,a+b)[:1])
    def test_gate_exact(self):
        rows=[dict(target=1,gt_id=i,scene=str(i//2),gt_area=10 if i<8 else 100) for i in range(32)]
        rows += [dict(target=0,gt_id=-1,scene=str(i//4),gt_area=2000) for i in range(128)]
        self.assertTrue(space_gate(rows)[0]);self.assertFalse(space_gate(rows[:-1])[0])

def prep():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not Path('/dev/nvidiactl').exists()
    cutoff(PREP_END);mark('prep')
    suite=unittest.TestSuite()
    for name in unittest.defaultTestLoader.getTestCaseNames(old.Contracts):
        if name!='test_budget':suite.addTest(old.Contracts(name))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName('test_m2f_tiny_verifier'))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    tests=unittest.TextTestRunner(verbosity=2).run(suite);assert tests.wasSuccessful() and not tests.skipped
    inputs={}
    def bind(p,h=None):
        p=Path(p);v=sha(p)
        if h is not None:assert v==h,str(p)
        inputs[str(p.relative_to(ROOT))]=v;return v
    bind(PRIOR/'report.json','cb97cde18e41fd70d8d6ba112de8f10329e721557c86d9b47d73a78bc40a2854')
    prior=read(PRIOR/'report.json');bind(PRIOR/'prepared_manifest.json',prior['artifacts_sha256']['prepared_manifest.json'])
    for n,h in read(PRIOR/'plan.json')['source_sha256'].items():bind(OUT/'source'/n,h)
    bind(old.BEST/'report.json',old.BEST_SHA);best=read(old.BEST/'report.json')
    bind(old.BEST/'binding.json',best['artifacts_sha256']['binding.json']);bind(old.BEST/'verifier_2000.pth',old.HEAD_SHA)
    bestbind=read(old.BEST/'binding.json')['input_sha256']
    bind(old.SCENE/'prepared_manifest.json',bestbind[str((old.SCENE/'prepared_manifest.json').relative_to(ROOT))])
    bind(old.OLD/'prepared_manifest.json','f33dc02cff69ca6fd6f63cadd8332f4ad62aa0392fc9bf23f913054e1532e243')
    bind(AUDIT/'report.json','74ec2c194aac53071a7697b7e887b3af1d048552e8e117d59146bf0882d5b54e')
    bind(AUDIT/'binding.json',read(AUDIT/'report.json')['artifacts_sha256']['binding.json'])
    auditbind=read(AUDIT/'binding.json')['input_sha256']
    previous=read(PRIOR/'prepared_manifest.json');sm=read(old.SCENE/'prepared_manifest.json');om=read(old.OLD/'prepared_manifest.json')
    excluded=set(previous['excluded_scenes'])|set(previous['scenes'].values());assert len(excluded)==958
    data=ROOT/w.base.DATA/'annotations/instances_train.json';bind(data,w.base.TRAIN_SHA)
    images=read(data)['images'];im={r['id']:r for r in images};assert len(images)==14432
    m=old.select_images(images,excluded,20261022);m['fit_ids']=sm['old_fit_ids'];assert len(m['fit_ids'])==384
    m['fit_scenes']={str(i):str(im[i]['scene_id']) for i in m['fit_ids']}
    assert not set(m['fit_scenes'].values())&set(m['scenes'].values())
    m['requests']={str(i):w.base.image_request(im[i],'train') for i in m['ids']+m['fit_ids']}
    m['input_hashes']={str(i):dict(image_id=i,rgb_sha256=bind(q['img_path']),depth_sha256=bind(q['depth_file'])) for i,q in ((int(k),q) for k,q in m['requests'].items())}
    for i in m['fit_ids']:
        p=w.REFERENCE/'current_train'/f'{i}.json';bind(p,auditbind[str(p.relative_to(ROOT))])
    parent=ROOT/'outputs/course_m2f_candidate_reuse_20261020_v1';cm=read(parent/'compact_manifest.json')
    for n in ('compact.json','live_config.py'):bind(w.PARENT/n,cm['files'][n])
    bind(w.PARENT/'selector.json',read(old.BEST/'binding.json')['selector_sha256']);bind(w.BEST/'refiner_3000.pth',w.HEAD_SHA)
    _,accepted,tiles,compact,cfg,loader,tail=old.setup_pipeline();bind(Path(compact['weight']),tiles.SPEC['weight_sha'])
    for n,h in read(ROOT/'outputs/course_m2f_tiles_20261019_v1/plan.json')['upstream_sha256'].items():bind(ROOT/'mmdetection'/n,h)
    import torch
    raw=loader(copy.deepcopy(m['requests'][str(m['ids'][0])]))
    for xy in old.WINDOWS:
        v=tail(old.crop512(raw,xy));assert tuple(v['inputs'].shape)==(5,1536,1536) and bool(torch.isfinite(v['inputs']).all())
    save(OUT/'prepared_manifest.json',m);save(OUT/'binding.json',dict(input_sha256=inputs,best_head_sha256=old.HEAD_SHA))
    save(OUT/'cpu_acceptance.json',dict(status='passed',tests=tests.testsRun,skipped=0,fit_images=384,check_images=128,excluded_scenes=958,gt_selected=False,full_val=False,unix=time.time()))

def fit_capture():
    import torch
    from mmengine.dataset import pseudo_collate
    from mmdet.registry import MODELS
    torch=w.base.configure_torch();assert torch.cuda.device_count()==1
    end=read(OUT/'phase_deadlines.json')['capture_end'];cutoff(end)
    original,accepted,tiles,compact,cfg,loader,tail=old.setup_pipeline()
    saved=torch.load(compact['weight'],map_location='cpu');model=MODELS.build(cfg.model)
    model.load_state_dict(saved['state_dict'],strict=True);del saved
    model.cuda().eval().requires_grad_(False);digest=original.tensor_digest(model.state_dict());calls={'eval':0,'train':0}
    def hook(module,args):calls['train' if module.training else 'eval']+=1
    handle=model.register_forward_pre_hook(hook);m=read(OUT/'prepared_manifest.json');files=[]
    (OUT/'fit_captures').mkdir();started=time.time();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for n,ident in enumerate(m['fit_ids']):
            assert time.time()<end;request=m['requests'][str(ident)]
            assert dict(image_id=ident,rgb_sha256=sha(request['img_path']),depth_sha256=sha(request['depth_file']))==m['input_hashes'][str(ident)]
            raw=loader(copy.deepcopy(request));rows=[];origins=[]
            for view,xy in enumerate(old.WINDOWS,5):
                v=tail(old.crop512(raw,xy));ds=v['data_samples']
                if 'gt_instances' in ds:assert len(ds.gt_instances)==0;del ds.gt_instances
                assert tuple(v['inputs'].shape)==(5,1536,1536) and tuple(ds.ori_shape)==(512,512)
                pred=model.test_step(pseudo_collate([v]))[0].cpu();r,s=old.encode512(pred.pred_instances,xy,ident,view)
                rows.extend(r);origins.extend(s);del v,pred
            p=OUT/'fit_captures'/f'{ident}.json';save(p,dict(image_id=ident,new_raw=rows,new_origins=origins,input_hashes=m['input_hashes'][str(ident)]))
            files.append(dict(image_id=ident,sha256=sha(p)));mark('fit_capture',images=n+1,expected=384,eval_calls=calls['eval'])
            if n==15:
                # Fresh128 has already completed; reserve15% on measured fit throughput.
                projected=(time.time()-started)/16*368*1.15+30
                save(OUT/'throughput_fit_gate.json',dict(projected_seconds=projected,end=end,unix=time.time()))
                assert time.time()+projected<end,'Cannot finish fixed384+128 in reserved GPU budget'
    handle.remove();assert calls=={'eval':1536,'train':0};assert original.tensor_digest(model.state_dict())==digest
    save(OUT/'fit_capture_report.json',dict(status='passed',files=files,calls=calls,state_before=digest,state_after=digest,seconds=time.time()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    del model;torch.cuda.empty_cache()

def capture():
    old.capture();fit_capture()
    a=read(OUT/'fit_capture_report.json');b=read(OUT/'capture_report.json')
    assert a['calls']['eval']+b['calls']['eval']==2688 and a['calls']['train']+b['calls']['train']==0
    save(OUT/'combined_capture.json',dict(status='passed',eval_calls=2688,train_calls=0,fit=sha(OUT/'fit_capture_report.json'),check=sha(OUT/'capture_report.json')))

def make_patches():
    import numpy as np
    import m2f_tiny_verifier as tiny
    import m2f_boundary_refiner as ref
    from pycocotools import mask as mu
    cutoff(LABEL_END);assert not Path('/dev/nvidiactl').exists();m=read(OUT/'prepared_manifest.json');coco=w.coco_load('train');records=[]
    fit=read(OUT/'fit_capture_report.json');(OUT/'check_patches').mkdir()
    for n,file in enumerate(fit['files']):
        i=file['image_id'];p=OUT/'fit_captures'/f'{i}.json';assert sha(p)==file['sha256'];rows=read(p)['new_raw']
        current=read(w.REFERENCE/'current_train'/f'{i}.json')
        if isinstance(current,dict):current=current['rows']
        jobs=tiny.eligible(current,rows);anns=[a for a in coco.loadAnns(coco.getAnnIds(imgIds=[i])) if not a.get('iscrowd',0) and not a.get('ignore',0)]
        for job in jobs:
            r=rows[job['index']];aa=[a for a in anns if a['category_id']==r['category_id']]
            values=mu.iou([r['segmentation']],[coco.annToRLE(a) for a in aa],[0]*len(aa))[0] if aa else []
            k=int(np.argmax(values)) if len(values) else None;a=aa[k] if k is not None else None;v=float(values[k]) if k is not None else 0.
            target=tiny.target(v,a['area'] if a else None)
            records.append(dict(image_id=i,scene=m['fit_scenes'][str(i)],**job,target=target,gt_id=a['id'] if a else None,gt_area=a['area'] if a else None,iou=v))
        mark('labels',images=n+1,expected=384)
    passed,counts=space_gate(records);save(OUT/'fit_labels.json',records)
    save(OUT/'space_gate.json',dict(status='passed' if passed else 'stopped',counts=counts,passed=passed,unix=time.time()))
    if not passed:return
    valid=[r for r in records if r['target'] in (0,1)];save(OUT/'train_records.json',valid)
    x=np.lib.format.open_memmap(OUT/'train_patches.npy',mode='w+',dtype=np.float32,shape=(len(valid),6,128,128))
    by={}
    for k,r in enumerate(valid):by.setdefault(r['image_id'],[]).append((k,r))
    for n,(i,group) in enumerate(by.items()):
        rgbd=ref.load_rgbd(m['requests'][str(i)]);rows=read(OUT/'fit_captures'/f'{i}.json')['new_raw']
        for k,r in group:x[k]=tiny.patch(rows[r['index']],r['xy'],rgbd)
        x.flush();mark('fit_patches',images=n+1,expected=len(by))
    del x
    # No GT is used to choose candidates or create inference tensors.
    sets=[('check',OUT,m['ids'],m['requests']),('continuity',PRIOR,read(PRIOR/'prepared_manifest.json')['ids'],read(PRIOR/'prepared_manifest.json')['requests'])]
    patchfiles=[]
    for group,root,ids,requests in sets:
        manifest=read(root/'capture_report.json');expected={r['image_id']:r['sha256'] for r in manifest['files']}
        for n,i in enumerate(ids):
            p=root/'captures'/f'{i}.json';assert sha(p)==expected[i];d=read(p);rgbd=ref.load_rgbd(requests[str(i)])
            actual=dict(image_id=i,rgb_sha256=sha(requests[str(i)]['img_path']),depth_sha256=sha(requests[str(i)]['depth_file']));assert actual==d['input_hashes']
            jobs=d['after_candidates'][len(d['before_candidates']):];joint=d['merged']+d['new_raw']
            a=np.empty((len(jobs),6,128,128),np.float32)
            for k,r in enumerate(jobs):a[k]=tiny.patch(joint[r['index']],r['xy'],rgbd)
            path=OUT/'check_patches'/f'{group}_{i}.npy';np.save(path,a,allow_pickle=False)
            patchfiles.append(dict(group=group,image_id=i,file=str(path.relative_to(OUT)),sha256=sha(path),capture=str(p),capture_sha256=sha(p)))
            mark('check_patches',group=group,images=n+1,expected=128)
    save(OUT/'patch_manifest.json',dict(train_sha256=sha(OUT/'train_patches.npy'),records_sha256=sha(OUT/'train_records.json'),files=patchfiles,fit_examples=len(valid),no_sample_dropped=True))

def traincheck():
    import numpy as np
    import torch
    import m2f_tiny_verifier as tiny
    import run_m2f_tiny_hflip as flip
    import run_m2f_quality_select as original
    end=read(OUT/'phase_deadlines.json')['train_end'];cutoff(end);torch=w.base.configure_torch()
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    assert torch.cuda.device_count()==1 and sha(old.BEST/'verifier_2000.pth')==old.HEAD_SHA
    torch.manual_seed(20261013);np.random.seed(20261013)
    head=tiny.Verifier();head.load_state_dict(torch.load(old.BEST/'verifier_2000.pth',map_location='cpu')['state_dict'],strict=True)
    head.cuda().train();initial={k:v.clone() for k,v in head.state_dict().items()};initial_hash=original.tensor_digest(initial)
    pm=read(OUT/'patch_manifest.json');assert sha(OUT/'train_patches.npy')==pm['train_sha256'];assert sha(OUT/'train_records.json')==pm['records_sha256']
    data=np.load(OUT/'train_patches.npy',mmap_mode='r');records=read(OUT/'train_records.json');targets=np.array([r['target'] for r in records],np.float32)
    pos=np.where(targets==1)[0];neg=np.where(targets==0)[0];assert len(pos) and len(neg)
    rng=torch.Generator().manual_seed(20261013);aug=torch.Generator().manual_seed(20261021)
    opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001);history=[];started=time.time();flipped=0;samplehash=hashlib.sha256();fliphash=hashlib.sha256()
    for step in range(1,2001):
        assert time.time()<end
        idx=np.concatenate((pos[torch.randint(len(pos),(16,),generator=rng).numpy()],neg[torch.randint(len(neg),(16,),generator=rng).numpy()]))
        flags=torch.rand(32,generator=aug)<.5;flipped+=int(flags.sum());samplehash.update(idx.tobytes());fliphash.update(flags.numpy().tobytes())
        x=torch.from_numpy(np.array(data[idx],copy=True)).cuda();y=torch.from_numpy(targets[idx]).cuda()
        x=flip.augment(x,flags.cuda());opt.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(head(x),y)
        assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())
        opt.step();assert all(bool(torch.isfinite(p).all()) for p in head.parameters())
        history.append(dict(step=step,loss=float(loss.detach())))
        if step%100==0:mark('train',step=step,total=2000)
    native=sorted({int(v['step'].item()) for v in opt.state.values()});assert native==[2000]
    updated={k:not torch.equal(v,initial[k]) for k,v in head.state_dict().items()};assert all(updated.values())
    trainseconds=time.time()-started;head.eval().requires_grad_(False);weight=OUT/'verifier_512_2000.pth'
    torch.save(dict(state_dict={k:v.cpu() for k,v in head.state_dict().items()},optimizer=opt.state_dict(),steps=2000,initial_head_sha256=old.HEAD_SHA),weight)
    outputs=[]
    with torch.inference_mode():
        for n,f in enumerate(pm['files']):
            assert time.time()<end;assert sha(OUT/f['file'])==f['sha256'] and sha(f['capture'])==f['capture_sha256']
            datax=np.load(OUT/f['file'],mmap_mode='r');ps=[]
            for begin in range(0,len(datax),32):
                z=np.zeros((32,6,128,128),np.float32);b=datax[begin:begin+32];z[:len(b)]=b
                ps.extend(map(float,torch.sigmoid(head(torch.from_numpy(z).cuda())).cpu().numpy()[:len(b)]))
            d=read(f['capture']);oldps=[r['probability'] for r in d['before_candidates']];jobs=[{k:v for k,v in r.items() if k!='probability'} for r in d['after_candidates']]
            joint=d['merged']+d['new_raw'];rows,activity=tiny.merge(d['current'],joint,jobs,oldps+ps)
            # Original probabilities and baseline output are explicitly reproduced.
            oldjobs=[{k:v for k,v in r.items() if k!='probability'} for r in d['before_candidates']]
            assert tiny.merge(d['current'],d['merged'],oldjobs,oldps)==(d['before'],d['before_activity'])
            outputs.append(dict(group=f['group'],image_id=f['image_id'],rows=rows,activity=activity,new_probabilities=ps,old_probabilities=oldps,capture=f['capture']))
            mark('head_check',images=n+1,expected=256)
    assert sha(old.BEST/'verifier_2000.pth')==old.HEAD_SHA
    save(OUT/'head_outputs.json',outputs);save(OUT/'train_history.json',history)
    save(OUT/'train_report.json',dict(status='passed',native_steps=native,finite_steps=2000,examples=64000,updated=updated,initial_state_sha256=initial_hash,initial_weight_sha256=old.HEAD_SHA,final_weight_sha256=sha(weight),train_seconds=trainseconds,flipped=flipped,sample_sha256=samplehash.hexdigest(),flip_sha256=fliphash.hexdigest(),old_head_unchanged=True,new_head_parameters=24369,check_images=256,unix=time.time()))

def finish():
    cutoff(ANALYSIS_END);assert not Path('/dev/nvidiactl').exists()
    gate=read(OUT/'space_gate.json');outcome='stopped_no_fit_space';result={}
    if gate['passed']:
        coco=w.coco_load('train');m=read(OUT/'prepared_manifest.json');outs=read(OUT/'head_outputs.json')
        for group in ('check','continuity'):
            chosen=[o for o in outs if o['group']==group];ids=[o['image_id'] for o in chosen];arms={k:[] for k in ('best5','frozen9','adapted9')}
            for o in chosen:
                d=read(o['capture']);arms['best5'].extend(d['before']);arms['frozen9'].extend(d['after']);arms['adapted9'].extend(o['rows'])
            metrics={k:w.metrics(coco,ids,v) for k,v in arms.items()};matched={k:w.matched(coco,ids,v) for k,v in arms.items()}
            summary={k:dict(small=len(v['small']),tiny=len(v['tiny']),false_positives=len(v['false_positives'])) for k,v in matched.items()}
            transfers={k:dict(tiny_gained=sorted(set(matched[k]['tiny'])-set(matched['best5']['tiny'])),tiny_lost=sorted(set(matched['best5']['tiny'])-set(matched[k]['tiny'])),small_gained=sorted(set(matched[k]['small'])-set(matched['best5']['small'])),small_lost=sorted(set(matched['best5']['small'])-set(matched[k]['small']))) for k in ('frozen9','adapted9')}
            result[group]=dict(metrics=metrics,matching=summary,transfers=transfers,image_ids=ids,full_val=False)
            save(OUT/(group+'_matching.json'),matched)
        # metric keys are asserted rather than relying on index assumptions.
        mm=result['check']['metrics'];apskey=next(k for k in mm['best5'] if k in ('APs','segm_mAP_s','coco/segm_mAP_s'))
        positive=mm['adapted9'][apskey]>mm['best5'][apskey];tinygood=result['check']['matching']['adapted9']['tiny']>=result['check']['matching']['best5']['tiny']
        result['internal_gate']=dict(aps_strict_positive=positive,tiny_net_nonnegative=tinygood,passed=positive and tinygood)
        outcome='pilot_passed_request_fullval_budget' if positive and tinygood else 'stopped_failed_internal_gate'
    save(OUT/'comparison.json',result)
    artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.pth') and p.name not in ('status.json','progress.json','plan.json','launch.json','report.json','terminal_acceptance.json','phases.json')}
    save(OUT/'report.json',dict(status='passed',outcome=outcome,space_gate=gate,comparison=result,official_best_aps=old.BEST_APS,official_best_unchanged=True,full_val=False,m2f_updates=0,maskrcnn_paused=True,artifacts_sha256=artifacts,finished_unix=time.time()))

if __name__=='__main__':
    import sys
    {'prep':prep,'capture':capture,'labels':make_patches,'traincheck':traincheck,'finish':finish}[sys.argv[1]]()
