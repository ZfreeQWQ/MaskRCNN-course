"""Frozen source-specific512 verifier: one complete val, no optimization."""
import copy,json,math,os,statistics,time,unittest
from pathlib import Path
import run_m2f_fine512 as old
import run_m2f_tiny_positive as w

START=1792532008
PREP_END,GPU_END,ANALYSIS_END,DEADLINE=(START+n for n in (1200,6900,8100,9000))
NAME='course_m2f_fine512_fullval_20261021_v1'
OWNER='91d985fa1e8f45b591428e755ed2e6a1'
ROOT=Path('/workspace');OUT=ROOT/'outputs'/NAME
ADAPT=ROOT/'outputs/course_m2f_fine512_adapt_20261021_v2'
ADAPT_REPORT='e1d6fa8d6b30514771fa7efec17649b51d83f65be70b2b93fb0e73ab612e8a53'
ADAPT_WEIGHT='29dee6d8d71e783ec7dffc7ebe190152b7475eeed635dbf312a845edfeb3ae0b'
read,save,sha=w.read,w.save,w.sha
old.START,old.DEADLINE=START,DEADLINE

def cutoff(end):old.cutoff(end)
def mark(phase,**kw):save(OUT/'progress.json',dict(phase=phase,unix=time.time(),**kw))
def split_rows(rows,ids):
    result={i:[] for i in ids}
    for row in rows:result[row['image_id']].append(row)
    assert all(len(x)<=100 for x in result.values())
    return result
def jobs_without_p(rows):return [{k:v for k,v in r.items() if k!='probability'} for r in rows]

class Contracts(unittest.TestCase):
    def test_budget_calls(self):
        self.assertEqual(DEADLINE-START,9000);self.assertEqual(2584*4+(8+5+20)*9,10633)
    def test_source_split(self):
        x=[dict(index=2,probability=.2),dict(index=9,probability=.8)]
        self.assertEqual(jobs_without_p(x),[dict(index=2),dict(index=9)])
        self.assertEqual(x[0]['probability'],.2)
    def test_unchanged_merge(self):
        import m2f_tiny_verifier as tiny
        from test_m2f_tiny_verifier import row
        cur=[row()];r=[row(300)];j=tiny.eligible(cur,r)
        self.assertEqual(tiny.merge(cur,r,j,[.8]),tiny.merge(copy.deepcopy(cur),copy.deepcopy(r),j,[.8]))
    def test_exact_partial_source(self):
        import m2f_tiny_verifier as tiny
        from test_m2f_tiny_verifier import row
        cur=[row()];before=[row(300)];new=[row(500)]
        a=tiny.eligible(cur,before);b=tiny.eligible(cur,before+new)
        self.assertEqual(a,b[:len(a)]);self.assertGreaterEqual(b[-1]['index'],len(before))

def prep():
    import m2f_tiny_verifier as tiny
    cutoff(PREP_END);assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not Path('/dev/nvidiactl').exists()
    suite=unittest.TestSuite()
    for name in unittest.defaultTestLoader.getTestCaseNames(old.Contracts):
        if name!='test_budget':suite.addTest(old.Contracts(name))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName('test_m2f_tiny_verifier'))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    tests=unittest.TextTestRunner(verbosity=2).run(suite);assert tests.wasSuccessful() and not tests.skipped
    inputs={}
    def bind(path,expected=None):
        path=Path(path);h=sha(path)
        if expected is not None:assert h==expected,str(path)
        inputs[str(path.relative_to(ROOT))]=h;return h
    bind(ADAPT/'report.json',ADAPT_REPORT);ar=read(ADAPT/'report.json')
    assert ar['comparison']['internal_gate']['passed'] and read(ADAPT/'status.json')['status']=='complete'
    bind(ADAPT/'verifier_512_2000.pth',ADAPT_WEIGHT)
    bind(ADAPT/'plan.json');assert all(sha(OUT/'source'/n)==h for n,h in read(ADAPT/'plan.json')['source_sha256'].items())
    bind(old.BEST/'report.json',old.BEST_SHA);br=read(old.BEST/'report.json');assert br['new_aps']==old.BEST_APS
    for n in ('predictions.segm.json','full_val_metrics.json','val_activity.json','val_input_hashes.json','binding.json','verifier_2000.pth'):
        bind(old.BEST/n,br['artifacts_sha256'][n])
    bind(w.BEST/'report.json',w.BEST_SHA);cr=read(w.BEST/'report.json')
    for n in ('predictions.segm.json','val_input_hashes.json','refiner_3000.pth'):bind(w.BEST/n,cr['artifacts_sha256'][n])
    bind(w.PARENT/'source_binding.json','dff860edec79c2d6e4d5a4d388c2b6058ab25d56b22dfce9156a4ffb0a8a8860')
    for rel,h in read(w.PARENT/'source_binding.json')['full_val_cache_sha256'].items():bind(ROOT/rel,h)
    ab=read(ADAPT/'binding.json')['input_sha256']
    for path in (w.PARENT/'compact.json',w.PARENT/'live_config.py',w.PARENT/'selector.json'):
        bind(path,ab[str(path.relative_to(ROOT))])
    compact=read(w.PARENT/'compact.json');bind(Path(compact['weight']),ab[str(Path(compact['weight']).relative_to(ROOT))])
    ann=ROOT/w.base.DATA/'annotations/instances_val.json';bind(ann)
    coco=w.coco_load('val');metrics=read(old.BEST/'full_val_metrics.json');ids=metrics['image_ids']
    assert len(ids)==len(set(ids))==2584 and set(ids)==set(coco.imgs)
    expected={x['image_id']:x for x in read(old.BEST/'val_input_hashes.json')}
    assert read(old.BEST/'val_input_hashes.json')==read(w.BEST/'val_input_hashes.json')
    before=split_rows(read(old.BEST/'predictions.segm.json'),ids)
    current=split_rows(read(w.BEST/'predictions.segm.json'),ids)
    merged=split_rows(read(ROOT/'outputs/course_m2f_tiles_20261019_v1/evaluate/predictions.segm.json'),ids)
    global_rows=split_rows(read(ROOT/'outputs/course_m2f_copypaste1536_20261018_v1/evaluate_1536/predictions.segm.json'),ids)
    activities={a['image_id']:a for a in read(old.BEST/'val_activity.json')};files=[]
    (OUT/'base_cache').mkdir()
    for n,i in enumerate(ids):
        assert time.time()<PREP_END;a=activities[i];jobs=jobs_without_p(a['candidates']);ps=[j['probability'] for j in a['candidates']]
        assert tiny.eligible(current[i],merged[i])==jobs
        final,activity=tiny.merge(current[i],merged[i],jobs,ps)
        assert final==before[i] and activity=={k:v for k,v in a.items() if k not in ('candidates','image_id')}
        p=OUT/'base_cache'/f'{i}.json';save(p,dict(image_id=i,global_rows=global_rows[i],current=current[i],merged=merged[i],before=before[i],before_activity=activity,jobs=jobs,probabilities=ps,input_hashes=expected[i],request=w.base.image_request(coco.imgs[i],'val')))
        files.append(dict(image_id=i,sha256=sha(p)))
        if n%64==0:mark('cpu_cache_parity',images=n+1,expected=2584)
    save(OUT/'prepared_manifest.json',dict(ids=ids,files=files,all_2584_old_outputs_exact=True,gt_to_model=False))
    save(OUT/'binding.json',dict(input_sha256=inputs,baseline_metrics=metrics['metrics'],baseline_report_sha256=old.BEST_SHA,adapt_report_sha256=ADAPT_REPORT,adapt_weight_sha256=ADAPT_WEIGHT))
    save(OUT/'cpu_acceptance.json',dict(status='passed',tests=tests.testsRun,skipped=0,exact_old_outputs=2584,old_ap_recomputed=False,unix=time.time()))

def gpu():
    import numpy as np
    import torch
    import m2f_boundary_refiner as ref
    import m2f_quality_selector as quality
    import m2f_tiny_verifier as tiny
    from mmengine.dataset import pseudo_collate
    from mmdet.registry import MODELS
    end=read(OUT/'phase_deadlines.json')['gpu_end'];cutoff(end)
    torch=w.base.configure_torch();assert torch.cuda.device_count()==1
    original,accepted,tiles,compact,cfg,loader,tail=old.setup_pipeline()
    binding=read(OUT/'binding.json')
    for path in (Path(compact['weight']),w.BEST/'refiner_3000.pth',old.BEST/'verifier_2000.pth',ADAPT/'verifier_512_2000.pth'):
        assert sha(path)==binding['input_sha256'][str(path.relative_to(ROOT))]
    ck=torch.load(compact['weight'],map_location='cpu');m2f=MODELS.build(cfg.model);m2f.load_state_dict(ck['state_dict'],strict=True);del ck
    m2f.cuda().eval().requires_grad_(False);correction=w.load_current().cuda().eval().requires_grad_(False)
    def head(path):
        h=tiny.Verifier();h.load_state_dict(torch.load(path,map_location='cpu')['state_dict'],strict=True);return h.cuda().eval().requires_grad_(False)
    oldhead=head(old.BEST/'verifier_2000.pth');newhead=head(ADAPT/'verifier_512_2000.pth')
    models=dict(m2f=m2f,correction=correction,old_verifier=oldhead,new_verifier=newhead)
    state={k:original.tensor_digest(v.state_dict()) for k,v in models.items()};calls=dict(eval=0,train=0)
    def hook(m,args):calls['train' if m.training else 'eval']+=1
    handle=m2f.register_forward_pre_hook(hook);selector=read(w.PARENT/'selector.json')
    def m2f_flags():
        torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.deterministic=False
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.benchmark
    def head_flags():torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    def new_views(request,i):
        raw=loader(copy.deepcopy(request));new=[];origins=[];m2f_flags()
        for view,xy in enumerate(old.WINDOWS,5):
            assert time.time()<end
            v=tail(old.crop512(raw,xy));ds=v['data_samples']
            if 'gt_instances' in ds:assert len(ds.gt_instances)==0;del ds.gt_instances
            assert tuple(v['inputs'].shape)==(5,1536,1536) and tuple(ds.ori_shape)==(512,512) and bool(torch.isfinite(v['inputs']).all())
            pred=m2f.test_step(pseudo_collate([v]))[0].cpu()
            rows,sources=old.encode512(pred.pred_instances,xy,i,view);new.extend(rows);origins.extend(sources)
            del pred,v
        return new,origins
    def combine(current,merged,jobs,ps,new,rgbd):
        head_flags();joint=merged+new;alljobs=tiny.eligible(current,joint)
        assert alljobs[:len(jobs)]==jobs
        newps=old.probabilities(alljobs[len(jobs):],joint,rgbd,newhead)
        rows,activity=tiny.merge(current,joint,alljobs,ps+newps)
        assert len(rows)<=100
        return rows,dict(activity,candidates=[dict(j,probability=p) for j,p in zip(alljobs,ps+newps)])
    def actual_full(cache):
        assert time.time()<end;torch.cuda.synchronize();t=time.perf_counter();request=cache['request'];i=cache['image_id']
        m2f_flags();views=tiles.pack_views(request,loader,tail);encoder=accepted.CandidateEncoder(tiles);pool=[];globals_=[]
        for k,v in enumerate(views):
            pred=m2f.test_step(pseudo_collate([v]))[0].cpu();rows,cs=encoder.encode(pred.pred_instances,k,i)
            if k==0:globals_=rows
            pool.extend(cs);del pred
        merged=accepted.base.direct_rows(tiles.merge_candidates(pool),i);chosen=quality.choose(globals_,merged,selector)[0]
        head_flags();rgbd=ref.load_rgbd(request);current=ref.infer(chosen,rgbd,correction,'cuda')[0]
        jobs=tiny.eligible(current,merged);ps=old.probabilities(jobs,merged,rgbd,oldhead)
        new,origins=new_views(request,i);final,activity=combine(current,merged,jobs,ps,new,rgbd)
        json.dumps(final,separators=(',',':'),allow_nan=False);torch.cuda.synchronize();ms=(time.perf_counter()-t)*1000
        assert encoder.encoded==encoder.roundtrips
        return dict(global_rows=globals_,current=current,merged=merged,jobs=jobs,probabilities=ps,new_raw=new,new_origins=origins,rows=final,activity=activity),ms
    manifest=read(OUT/'prepared_manifest.json');ids=manifest['ids'];files=[];predictions=[];inputs=[];start=time.time()
    (OUT/'captures').mkdir();(OUT/'predictions').mkdir()
    with torch.inference_mode():
        for n,item in enumerate(manifest['files']):
            assert time.time()<end;i=item['image_id'];p=OUT/'base_cache'/f'{i}.json';assert sha(p)==item['sha256'];cache=read(p);request=cache['request']
            ih=dict(image_id=i,rgb_sha256=sha(request['img_path']),depth_sha256=sha(request['depth_file']));assert ih==cache['input_hashes'];inputs.append(ih)
            new,origins=new_views(request,i);rgbd=ref.load_rgbd(request)
            final,activity=combine(cache['current'],cache['merged'],cache['jobs'],cache['probabilities'],new,rgbd)
            assert tiny.merge(cache['current'],cache['merged'],cache['jobs'],cache['probabilities'])==(cache['before'],cache['before_activity'])
            capture=OUT/'captures'/f'{i}.json';save(capture,dict(image_id=i,new_raw=new,new_origins=origins,rows=final,activity=activity,input_hashes=ih))
            output=OUT/'predictions'/f'{i}.json';save(output,final);predictions.extend(final)
            files.append(dict(image_id=i,capture_sha256=sha(capture),prediction_sha256=sha(output)))
            if n%16==0:mark('new_views_val',images=n+1,expected=2584,eval_calls=calls['eval'],train_calls=calls['train'])
            if n==31:
                projected=(time.time()-start)/32*(2584-32)*1.15+240
                save(OUT/'throughput_gate.json',dict(status='passed' if time.time()+projected<end else 'failed',images=32,projected_remaining_seconds=projected,end=end,unix=time.time()))
                assert time.time()+projected<end,'Original GPU budget insufficient; no protocol reduction'
        assert calls==dict(eval=10336,train=0)
        save(OUT/'predictions.segm.json',predictions);save(OUT/'val_input_hashes.json',inputs)
        save(OUT/'predictions_frozen.json',dict(images=2584,image_ids=ids,files=files,prediction_sha256=sha(OUT/'predictions.segm.json'),unix=time.time()))
        del predictions
        parity=[]
        for i in w.base.FIXED8:
            cache=read(OUT/'base_cache'/f'{i}.json');saved=read(OUT/'captures'/f'{i}.json');actual,ms=actual_full(cache)
            for key in ('global_rows','current','merged','jobs','probabilities'):assert actual[key]==cache[key],(i,key)
            for key in ('new_raw','new_origins','rows','activity'):assert actual[key]==saved[key],(i,key)
            parity.append(dict(image_id=i,old_outputs_exact=True,new_raw_exact=True,probabilities_exact=True,final_instances_exact=True))
            mark('actual_parity',images=len(parity),expected=8)
        save(OUT/'parity.json',parity)
        cache=read(OUT/'base_cache'/f'{w.base.FIXED8[0]}.json');expected=read(OUT/'captures'/f'{w.base.FIXED8[0]}.json');times=[];peaks=[];reserved=[]
        for n in range(25):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();actual,ms=actual_full(cache)
            assert actual['rows']==expected['rows'] and actual['activity']==expected['activity']
            if n>=5:times.append(ms);peaks.append(torch.cuda.max_memory_allocated()/2**20);reserved.append(torch.cuda.max_memory_reserved()/2**20)
            mark('full_timing',completed=n+1,expected=25)
    handle.remove();assert calls==dict(eval=10633,train=0),calls
    after={k:original.tensor_digest(v.state_dict()) for k,v in models.items()};assert after==state
    save(OUT/'timing.json',dict(warmup=5,repeats=20,milliseconds=times,mean_ms=statistics.mean(times),median_ms=statistics.median(times),peak_allocated_mib=max(peaks),peak_reserved_mib=max(reserved),scope='RGB-D read/decode,nine1536views,M2F,D2H,RLE,merge,selection,4px correction,two-source verifier,NMS/top100,JSON; excludes output write/hash',paired_baseline=False))
    save(OUT/'gpu_report.json',dict(status='passed',calls=calls,state_before=state,state_after=after,old_weights_unchanged=True,new_head_unchanged=True,training_updates=0,full_old_m2f_val_repeated=False,images=2584,parity_images=8,active_parameters=m2f.active_parameter_count()+sum(p.numel() for k,m in models.items() if k!='m2f' for p in m.parameters()),model_state_bytes=sum(t.numel()*t.element_size() for m in models.values() for t in m.state_dict().values()),unix=time.time()))

def finish():
    import m2f_tiny_positive_helpers as h
    cutoff(ANALYSIS_END);assert not Path('/dev/nvidiactl').exists()
    assert read(OUT/'gpu_report.json')['status']=='passed' and len(read(OUT/'parity.json'))==8
    frozen=read(OUT/'predictions_frozen.json');assert sha(OUT/'predictions.segm.json')==frozen['prediction_sha256']
    ids=frozen['image_ids'];coco=w.coco_load('val');rows=read(OUT/'predictions.segm.json')
    assert len(ids)==len(set(ids))==2584 and set(ids)==set(coco.imgs) and not (OUT/'full_val_metrics.json').exists()
    metrics=w.metrics(coco,ids,rows);save(OUT/'full_val_metrics.json',dict(metrics=metrics,image_ids=ids,images=2584,evaluations=1))
    before=read(old.BEST/'predictions.segm.json');summary,matched=h.paired_summary(w,coco,ids,before,rows)
    summary['area_bins']={name:dict(gained=sum(lo<=coco.anns[i]['area']<=hi for i in summary['small_gained']),lost=sum(lo<=coco.anns[i]['area']<=hi for i in summary['small_lost'])) for name,lo,hi in [('1_16',1,16),('17_64',17,64),('65_256',65,256),('257_1024',257,1024)]}
    save(OUT/'full_val_error_analysis.json',summary);save(OUT/'new_matching.json',matched)
    baseline=read(OUT/'binding.json')['baseline_metrics'];comparison=dict(metrics=metrics,baseline=baseline,delta_points={k:100*(metrics[k]-v) for k,v in baseline.items()},any_positive_retained=metrics['APs']>old.BEST_APS,single_development_val=True,small_positive_not_rejected_by_desired_gain=True)
    save(OUT/'comparison.json',comparison)
    assert all(sha(ROOT/p)==v for p,v in read(OUT/'binding.json')['input_sha256'].items())
    artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix=='.json' and p.name not in ('status.json','progress.json','plan.json','launch.json','report.json','terminal_acceptance.json','phases.json')}
    save(OUT/'report.json',dict(status='passed',outcome='evaluated',comparison=comparison,baseline_aps=old.BEST_APS,new_aps=metrics['APs'],full_val_evaluations=1,model_updates=0,maskrcnn_paused=True,artifacts_sha256=artifacts,start_unix=START,deadline_unix=DEADLINE,finished_unix=time.time()))

if __name__=='__main__':
    import sys
    {'prep':prep,'gpu':gpu,'finish':finish}[sys.argv[1]]()
