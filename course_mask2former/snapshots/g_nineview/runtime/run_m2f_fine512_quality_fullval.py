"""Frozen selector, one official cached full-val evaluation, no fitting or GPU."""
import collections
import hashlib
import json
import os
import statistics
import time
import unittest
from pathlib import Path
import run_m2f_fine512_quality as prior
import m2f_fine512_quality as f
import m2f_quality_selector as q

START=1792573390
BIND_END=START+300
AP_END=START+1200
ANALYSIS_END=START+1500
DEADLINE=START+1800
NAME='course_m2f_fine512_quality_fullval_20261021_v1'
ROOT=Path('/workspace');OUT=ROOT/'outputs'/NAME
PILOT=ROOT/'outputs/course_m2f_fine512_quality_20261021_v1'
g=prior.g;w=prior.w
INPUTS={}

def cutoff(end=ANALYSIS_END):assert time.time()<end,'Frozen fullval absolute phase deadline'
def read(path,expected=None):
    path=Path(path);raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    assert expected is None or digest==expected,str(path)
    key=str(path.relative_to(ROOT));assert key not in INPUTS or INPUTS[key]==digest
    INPUTS[key]=digest;return json.loads(raw)
def save(n,d):w.save(OUT/n,d)
def mark(phase,**data):
    cutoff();v=dict(phase=phase,unix=time.time(),**data);save('progress.json',v);print(json.dumps(v),flush=True)
def forbidden_fit(*args,**kwargs):raise AssertionError('Refitting forbidden in frozen fullval')
def transfers(coco,before,after):
    out=dict(tiny_gained=sorted(set(after['tiny'])-set(before['tiny'])),
        tiny_lost=sorted(set(before['tiny'])-set(after['tiny'])),
        small_gained=sorted(set(after['small'])-set(before['small'])),
        small_lost=sorted(set(before['small'])-set(after['small'])),
        fp_before=len(before['false_positives']),fp_after=len(after['false_positives']))
    bins={}
    for lo,hi in ((1,16),(17,64),(65,256),(257,1024)):
        b={i for i in before['small'] if lo<=coco.anns[i]['area']<=hi}
        a={i for i in after['small'] if lo<=coco.anns[i]['area']<=hi}
        bins[f'{lo}-{hi}']=dict(before=len(b),after=len(a),gained=sorted(a-b),lost=sorted(b-a))
    out['area_bins']=bins;out['evicted_output_slots']=0;out['scope']='COCO small IoU.5 one-to-one, not AP'
    return out

def main():
    cutoff(BIND_END);assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not Path('/dev/nvidiactl').exists()
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName(n) for n in
        ('test_m2f_fine512_geometry','test_m2f_fine512_quality','test_m2f_fine512_quality_fullval')])
    tests=unittest.TextTestRunner(verbosity=2).run(suite);assert tests.wasSuccessful() and not tests.skipped
    q.fit=forbidden_fit  # No path can refit after synthetic contract fixtures.
    pr=read(PILOT/'report.json','791e65bc9f5704e20ff306d15d661cdb6184a6f0b230ae5a7e36c870c0d8b419')
    assert pr['outcome']=='stopped_failed_internal_gate' and pr['selector_fits']==1 and pr['full_val_evaluations']==0
    selector=read(PILOT/'selector.json','e5f3ef4e39d7f74a965a6d93301e279bbfda4d00f8964a89c5055f9b1ee41af8')
    assert selector['threshold']==0 and selector['lambda_l2']==1 and selector['trainable_parameters']==105
    gate=read(PILOT/'internal_gate.json',pr['artifacts_sha256']['internal_gate.json'])
    assert not gate['passed'] and all(v['delta_points']['APs']>0 for v in gate['groups'].values())
    fr=read(g.FULL/'report.json','89b80f5704d4328b41d53804024ffab68d78309fdad2b2f31edebbfd4116f8fa')
    br=read(g.BEST/'report.json','8a73c0241f429c23daad53a2f9cf5674bada77b2dae114fb38e26f586aefb9e6')
    assert br['new_aps']==g.BEST_APS
    manifest=read(g.FULL/'prepared_manifest.json',fr['artifacts_sha256']['prepared_manifest.json'])
    frozen=read(g.FULL/'predictions_frozen.json',fr['artifacts_sha256']['predictions_frozen.json'])
    capsha={x['image_id']:x['capture_sha256'] for x in frozen['files']}
    baseline=read(g.BEST/'predictions.segm.json',br['artifacts_sha256']['predictions.segm.json'])
    bm=read(g.BEST/'full_val_metrics.json',br['artifacts_sha256']['full_val_metrics.json'])['metrics']
    bh=read(g.BEST/'val_input_hashes.json',br['artifacts_sha256']['val_input_hashes.json'])
    val=w.coco_load('val');ids=manifest['ids']
    assert len(ids)==len(set(ids))==2584 and ids==frozen['image_ids'] and set(ids)==set(val.imgs)
    assert [x['image_id'] for x in manifest['files']]==ids
    save('cpu_acceptance.json',dict(status='passed',tests=tests.testsRun,zero_skip=True,selector_fits=0,
        original_gate_unchanged=True,new_authorization='Tiny per-group protection reported as risk; one final frozen fullval',
        original_source_sha_unchanged=True,GPU=False,baseline_APs=g.BEST_APS))
    cutoff(BIND_END);start=time.perf_counter();mark('fullval_cached_transform')
    after=[];before=[];hashes=[];records=[];audits=[];sample=None;counts=collections.Counter()
    for n,item in enumerate(manifest['files']):
        cutoff(AP_END);ident=item['image_id'];b=read(g.FULL/'base_cache'/f'{ident}.json',item['sha256'])
        d=read(g.FULL/'captures'/f'{ident}.json',capsha[ident])
        assert d['image_id']==b['image_id']==d['input_hashes']['image_id']==ident and b['input_hashes']==d['input_hashes']
        rows,activity=f.apply(b['current'],d['new_raw'],b['before'],b['before_activity']['kept'],selector)
        assert all(r['image_id']==ident and r['segmentation']['size']==[1024,1024] for r in rows)
        assert len(rows)==len(b['before'])<=100
        # GT is used only after the fixed prediction, solely for analysis.
        truth=f.supervised(b['current'],d['new_raw'],prior.annotations(val,ident),val)
        audits+=f.decision_audit(truth,activity['decisions'])
        before+=b['before'];after+=rows;hashes.append(b['input_hashes'])
        records.append(dict(image_id=ident,before=b['before'],after=rows,activity=activity))
        for k in ('accepted','rejected','protected_tiny_outputs','current_pairs','final_pairs'):counts[k]+=activity[k]
        if sample is None:sample=(b,d)
        if n%128==0:mark('fullval_cached_transform',images=n+1,expected=2584)
    assert before==baseline and hashes==bh
    transform_seconds=time.perf_counter()-start
    save('predictions.segm.json',after);save('val_input_hashes.json',hashes)
    save('val_activity.json',[dict(image_id=r['image_id'],**r['activity']) for r in records])
    save('val_decision_audit.json',dict(summary=prior.audit_summary(audits),cases=audits))
    times=[];b,d=sample
    for n in range(25):
        cutoff(AP_END);t=time.perf_counter();rows,_=f.apply(b['current'],d['new_raw'],b['before'],b['before_activity']['kept'],selector)
        json.dumps(rows,separators=(',',':'),allow_nan=False)
        if n>=5:times.append((time.perf_counter()-t)*1000)
    save('timing.json',dict(warmup=5,repeats=20,mean_ms=statistics.mean(times),milliseconds=times,
        scope='Loaded original/raw RLE pairing, fixed selector, mapped replacement and JSON; excludes IO/model/multi-view cost',new_gpu_memory=False))
    cutoff(AP_END);mark('one_fullval_AP');assert not (OUT/'full_val_metrics.json').exists()
    t=time.perf_counter();metrics=w.metrics(val,ids,after);ap_seconds=time.perf_counter()-t
    save('full_val_metrics.json',dict(metrics=metrics,images=2584,image_ids=ids,evaluations=1,seconds=ap_seconds))
    cutoff(AP_END);mark('paired_analysis')
    bmatch=w.matched(val,ids,baseline);amatch=w.matched(val,ids,after)
    save('new_matching.json',amatch);save('baseline_matching.json',bmatch)
    error=transfers(val,bmatch,amatch);save('full_val_error_analysis.json',error)
    save('fixed_old_tp_audit.json',prior.fixed_true_audit(records,bmatch,val))
    relevant=set(error['small_gained']+error['small_lost'])
    save('selected_cases.json',[p for p in audits if p['gt_id'] in relevant])
    comp=dict(metrics=metrics,baseline=bm,delta_points={k:100*(metrics[k]-bm[k]) for k in metrics},
        any_positive_retained=metrics['APs']>g.BEST_APS,repeated_development_val=True,
        tiny_protection_risk_reported=True,no_independent_generalization_claim=True)
    save('comparison.json',comp)
    assert w.sha(PILOT/'selector.json')=='e5f3ef4e39d7f74a965a6d93301e279bbfda4d00f8964a89c5055f9b1ee41af8'
    assert w.sha(PILOT/'report.json')=='791e65bc9f5704e20ff306d15d661cdb6184a6f0b230ae5a7e36c870c0d8b419'
    save('binding.json',dict(input_sha256=INPUTS,selector_frozen=True,old_gate_report_unchanged=True,all_input_hashes_match_baseline=True))
    cutoff();plan=w.read(OUT/'plan.json');assert all(w.sha(OUT/'source'/n)==h for n,h in plan['source_sha256'].items())
    artifacts={p.name:w.sha(p) for p in OUT.glob('*.json') if p.name not in
        ('plan.json','status.json','progress.json','launch.json','report.json','terminal_acceptance.json','command.json','container.json')}
    save('report.json',dict(status='passed',outcome='evaluated',start_unix=START,finished_unix=time.time(),deadline_unix=DEADLINE,
        source_artifacts_bound=True,buffer_used=False,baseline_APs=g.BEST_APS,comparison=comp,
        full_val_evaluations=1,baseline_AP_evaluations=0,selector_fits=0,model_loads=0,weight_updates=0,
        GPU=False,new_model_predictions=0,maskrcnn_paused=True,images=2584,activity=dict(counts),
        transform_with_analysis_seconds=transform_seconds,AP_seconds=ap_seconds,artifacts_sha256=artifacts,
        selector_sha256='e5f3ef4e39d7f74a965a6d93301e279bbfda4d00f8964a89c5055f9b1ee41af8',
        original_per_group_tiny_gate_failed_preserved=True,all_GTs_analysis_only=True))

if __name__=='__main__':main()
