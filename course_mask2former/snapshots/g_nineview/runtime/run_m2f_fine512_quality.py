"""One CPU ridge fit, two train checks, at most one official full-val AP."""
import collections
import json
import os
import statistics
import time
import unittest
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
import m2f_quality_selector as q
import m2f_fine512_quality as f
import run_m2f_fine512_geometry as g

START=1792572360  # Preparation counted conservatively from16:46 Beijing.
PREP_END=START+900
PILOT_END=START+1500
COMPUTE_END=START+2700
DEADLINE=START+3600
NAME='course_m2f_fine512_quality_20261021_v1'
ROOT=Path('/workspace');OUT=ROOT/'outputs'/NAME
g.START,g.PREP_END,g.PILOT_END,g.COMPUTE_END,g.DEADLINE,g.NAME,g.OUT=START,PREP_END,PILOT_END,COMPUTE_END,DEADLINE,NAME,OUT
# The inherited cutoff's default argument was defined at import time.
def cutoff(end=None):
    assert time.time() < (COMPUTE_END if end is None else end), 'Immutable quality phase deadline'
g.cutoff=cutoff
w=g.w;save=g.save;read=g.read_bound;mark=g.mark


def annotations(coco,ident):
    return coco.loadAnns(coco.getAnnIds(imgIds=[ident]))


def audit_summary(records):
    c=collections.Counter()
    for p in records:
        c['pairs']+=1;c['accepted' if p['accepted'] else 'rejected']+=1
        if p['target_delta']>0:c['beneficial_accepted' if p['accepted'] else 'beneficial_rejected']+=1
        if p['target_delta']<0:c['harmful_accepted' if p['accepted'] else 'harmful_rejected']+=1
        if p['accepted']:
            c['same_gt_crossings']+=p['threshold_crossings']
            if p['gt_area'] is not None and p['gt_area']<=256:c['tiny_same_gt_crossings']+=p['threshold_crossings']
    return dict(c)


def fixed_true_audit(records, matching, coco):
    truth={p['detection_id']:p for p in matching['true_predictions']};offset=0;cases=[];c=collections.Counter()
    for r in records:
        for d in r['activity']['decisions']:
            if not d['accepted']:continue
            j=d['output_index'];hit=truth.get(offset+j+1)
            if hit is None:continue
            old,new=r['before'][j],r['after'][j];ann=coco.anns[hit['annotation_id']]
            assert hit['image_id']==r['image_id'] and hit['score']==old['score']==new['score']
            a,b=[float(x) for x in mu.iou([old['segmentation'],new['segmentation']],[coco.annToRLE(ann)],[0])[:,0]]
            assert a>=.5
            cross=sum(b>=t for t in np.arange(.5,1.,.05))-sum(a>=t for t in np.arange(.5,1.,.05))
            c['changed_old_TP']+=1;c['improved' if b>a else ('worsened' if b<a else 'equal')]+=1;c['crossings']+=cross
            if old['score']>=.9:
                c['score90_gain75']+=a<.75<=b;c['score90_loss75']+=b<.75<=a
            cases.append(dict(image_id=r['image_id'],gt_id=ann['id'],gt_area=ann['area'],
                              score=old['score'],old_iou=a,new_iou=b,crossings=int(cross),**d))
        offset+=len(r['before'])
    return dict(counts={k:int(v) for k,v in c.items()},cases=cases,
                scope='Fixed original IoU.5 true prediction assignment, not new AP or high-threshold matching')


def finish(outcome,fullval=0,comp=None):
    cutoff();save('binding.json',dict(input_sha256=g.INPUTS,inference_receives_gt=False,
        fit_split='official_train_384',baseline_aps=g.BEST_APS,old_model_weights_updated=False))
    plan=w.read(OUT/'plan.json');assert all(w.sha(OUT/'source'/n)==h for n,h in plan['source_sha256'].items())
    arts={p.name:w.sha(p) for p in OUT.glob('*.json') if p.name not in
          ('plan.json','status.json','progress.json','launch.json','report.json','terminal_acceptance.json','command.json','container.json')}
    save('report.json',dict(status='passed',outcome=outcome,start_unix=START,finished_unix=time.time(),
        deadline_unix=DEADLINE,buffer_used=False,full_val_evaluations=fullval,comparison=comp,
        selector_fits=int((OUT/'selector.json').exists()),selector_parameters=105,
        model_loads=0,old_weight_updates=0,gpu_used=False,new_model_predictions=0,
        artifacts_sha256=arts,maskrcnn_paused=True,checks_are_reused_development_train=True,
        baseline_aps=g.BEST_APS,measurement_scope='CPU postprocessing only; historical nineview cost remains'))


def main():
    cutoff(PREP_END);assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not Path('/dev/nvidiactl').exists()
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName(n) for n in
                           ('test_m2f_fine512_geometry','test_m2f_fine512_quality')])
    tests=unittest.TextTestRunner(verbosity=2).run(suite);assert tests.wasSuccessful() and not tests.skipped
    ar=read(g.ADAPT/'report.json','e1d6fa8d6b30514771fa7efec17649b51d83f65be70b2b93fb0e73ab612e8a53')
    rr=read(g.RAM/'report.json','432333b3e2f5db30955c4c33df54cf8b0a545169a5ba1993ee75c88ad86a67e2')
    fr=read(g.FULL/'report.json','89b80f5704d4328b41d53804024ffab68d78309fdad2b2f31edebbfd4116f8fa')
    br=read(g.BEST/'report.json','8a73c0241f429c23daad53a2f9cf5674bada77b2dae114fb38e26f586aefb9e6')
    assert br['new_aps']==g.BEST_APS
    manifest=read(g.ADAPT/'prepared_manifest.json',ar['artifacts_sha256']['prepared_manifest.json'])
    binding=read(g.ADAPT/'binding.json',ar['artifacts_sha256']['binding.json'])['input_sha256']
    fitfiles=read(g.ADAPT/'fit_capture_report.json',ar['artifacts_sha256']['fit_capture_report.json'])['files']
    fitsha={x['image_id']:x['sha256'] for x in fitfiles}
    pm=read(g.ADAPT/'patch_manifest.json',ar['artifacts_sha256']['patch_manifest.json'])
    past=read(g.RAM/'internal_gate.json',rr['artifacts_sha256']['internal_gate.json'])
    previous={k:read(g.ADAPT/(k+'_matching.json'),ar['artifacts_sha256'][k+'_matching.json'])['best5'] for k in ('check','continuity')}
    old=ROOT/'outputs/course_m2f_fine512_20261021_v1'
    oldr=read(old/'report.json','cb97cde18e41fd70d8d6ba112de8f10329e721557c86d9b47d73a78bc40a2854')
    oldm=read(old/'prepared_manifest.json',oldr['artifacts_sha256']['prepared_manifest.json'])
    fitids=manifest['fit_ids'];checkids=past['groups']['check']['ids'];contids=past['groups']['continuity']['ids']
    assert len(fitids)==len(set(fitids))==384 and checkids==manifest['ids'] and contids==oldm['ids']
    assert len(checkids)==len(contids)==128 and not set(fitids)&set(checkids+contids) and not set(checkids)&set(contids)
    scenes=[set(manifest['fit_scenes'].values()),set(manifest['scenes'].values()),set(oldm['scenes'].values())]
    assert all(not scenes[i]&scenes[j] for i in range(3) for j in range(i))
    coco=w.coco_load('train');assert set(fitids+checkids+contids)<=set(coco.imgs)
    pairs=[];start=time.perf_counter()
    for n,ident in enumerate(fitids):
        cutoff(PREP_END)
        p=w.REFERENCE/'current_train'/f'{ident}.json';current=read(p,binding[str(p.relative_to(ROOT))])
        d=read(g.ADAPT/'fit_captures'/f'{ident}.json',fitsha[ident])
        assert d['image_id']==d['input_hashes']['image_id']==ident
        pairs += [dict(p,scene=manifest['fit_scenes'][str(ident)]) for p in f.supervised(current,d['new_raw'],annotations(coco,ident),coco)]
        if n%32==0:mark('fit_pairs',images=n+1,expected=384,pairs=len(pairs))
    positive=[p for p in pairs if p['target_delta']>0];negative=[p for p in pairs if p['target_delta']<0]
    tinyspace={p['gt_id'] for p in positive if p['gt_area']<=256 and any(p['old_iou']<t<=p['new_iou'] for t in np.arange(.5,1.,.05))}
    passed=bool(positive and negative and tinyspace)
    save('fit_pairs.json',pairs);save('space_gate.json',dict(passed=passed,fit_images=384,pairs=len(pairs),
        positive=len(positive),negative=len(negative),identity_zero=sum(p['target_kind']=='identity_zero' for p in pairs),
        beneficial_tiny_crossing_gt=sorted(tinyspace),scene_disjoint=True,seconds=time.perf_counter()-start,
        gate='Both signs and at least one tiny same-GT crossing; GT never selects prediction'))
    save('cpu_acceptance.json',dict(status='passed',tests=tests.testsRun,zero_skip=True,source_and_scene_binding=True,
                                   fit_images=384,check_images=[128,128],GPU=False))
    if not passed:finish('stopped_no_fit_geometry_space');return
    cutoff(PILOT_END);mark('one_ridge_fit');start=time.perf_counter()
    selector=q.fit([p['features'] for p in pairs],[p['target_delta'] for p in pairs])
    save('selector.json',selector);selector_sha=w.sha(OUT/'selector.json')
    save('fit_report.json',dict(status='passed',fits=1,seconds=time.perf_counter()-start,selector_sha256=selector_sha,
                               diagnostic=q.diagnostic(pairs,selector),hyperparameter_search=False,fit_gt_used_only_in_labels=True))
    records=[];audit=[];sample=None
    for n,item in enumerate(pm['files']):
        cutoff(PILOT_END);d=read(item['capture'],item['capture_sha256']);ident=item['image_id'];group=item['group']
        assert d['image_id']==d['input_hashes']['image_id']==ident and ident in past['groups'][group]['ids']
        after,activity=f.apply(d['current'],d['new_raw'],d['before'],d['before_activity']['kept'],selector)
        # GT is used only AFTER prediction is complete.
        truth=f.supervised(d['current'],d['new_raw'],annotations(coco,ident),coco)
        audit += [dict(x,group=group) for x in f.decision_audit(truth,activity['decisions'])]
        records.append(dict(group=group,image_id=ident,before=d['before'],after=after,activity=activity,input_hashes=d['input_hashes']))
        if sample is None:sample=d
        if n%32==0:mark('internal_transform',images=n+1,expected=256)
    times=[]
    for n in range(25):
        t=time.perf_counter();rows,_=f.apply(sample['current'],sample['new_raw'],sample['before'],sample['before_activity']['kept'],selector)
        json.dumps(rows,separators=(',',':'),allow_nan=False)
        if n>=5:times.append((time.perf_counter()-t)*1000)
    save('timing.json',dict(warmup=5,repeats=20,mean_ms=statistics.mean(times),milliseconds=times,
        scope='Loaded current/raw RLE reciprocal pairing, ridge gain prediction, mapped replacement and JSON; no IO or model',new_gpu_memory=False))
    save('pilot_rows.json',records);save('decision_audit.json',dict(summary=audit_summary(audit),cases=audit))
    save('fixed_old_tp_audit.json',{k:fixed_true_audit([r for r in records if r['group']==k],previous[k],coco) for k in previous})
    groups={};allids=[];allafter=[]
    for group in ('check','continuity'):
        cutoff(PILOT_END);chosen=[r for r in records if r['group']==group];ids=[r['image_id'] for r in chosen]
        assert ids==past['groups'][group]['ids']
        before=[x for r in chosen for x in r['before']];after=[x for r in chosen for x in r['after']]
        groups[group],matching=g.comparison(coco,ids,before,after,past['groups'][group]['baseline'],previous[group])
        save(group+'_matching.json',matching);allids+=ids;allafter+=after
    combined=w.metrics(coco,allids,allafter);delta=100*(combined['APs']-past['combined_before']['APs'])
    passed=all(v['delta_points']['APs']>=0 and len(v['tiny_gained'])>=len(v['tiny_lost']) for v in groups.values()) and delta>0
    save('internal_gate.json',dict(passed=bool(passed),groups=groups,combined=combined,combined_before=past['combined_before'],
        combined_delta_aps_points=delta,no_baseline_AP_rerun=True,official_val=False))
    assert w.sha(OUT/'selector.json')==selector_sha
    if not passed:finish('stopped_failed_internal_gate');return
    mark('fullval_cached_transform')
    vm=read(g.FULL/'prepared_manifest.json',fr['artifacts_sha256']['prepared_manifest.json'])
    frozen=read(g.FULL/'predictions_frozen.json',fr['artifacts_sha256']['predictions_frozen.json'])
    caps={x['image_id']:x['capture_sha256'] for x in frozen['files']};ids=vm['ids'];val=w.coco_load('val')
    assert len(ids)==len(set(ids))==2584 and ids==frozen['image_ids'] and set(ids)==set(val.imgs)
    baseline=read(g.BEST/'predictions.segm.json',br['artifacts_sha256']['predictions.segm.json'])
    bm=read(g.BEST/'full_val_metrics.json',br['artifacts_sha256']['full_val_metrics.json'])['metrics']
    bh=read(g.BEST/'val_input_hashes.json',br['artifacts_sha256']['val_input_hashes.json'])
    before=[];after=[];hashes=[];activities=[];val_audit=[]
    for n,item in enumerate(vm['files']):
        cutoff();ident=item['image_id'];b=read(g.FULL/'base_cache'/f'{ident}.json',item['sha256']);d=read(g.FULL/'captures'/f'{ident}.json',caps[ident])
        assert b['input_hashes']==d['input_hashes'] and d['image_id']==ident
        rows,activity=f.apply(b['current'],d['new_raw'],b['before'],b['before_activity']['kept'],selector)
        before+=b['before'];after+=rows;hashes.append(b['input_hashes']);activities.append(dict(image_id=ident,**activity))
        truth=f.supervised(b['current'],d['new_raw'],annotations(val,ident),val)
        val_audit+=f.decision_audit(truth,activity['decisions'])
        if n%64==0:mark('fullval_cached_transform',images=n+1,expected=2584)
    assert before==baseline and hashes==bh and w.sha(OUT/'selector.json')==selector_sha
    save('predictions.segm.json',after);save('val_activity.json',activities);save('val_input_hashes.json',hashes)
    save('val_decision_audit.json',dict(summary=audit_summary(val_audit),cases=val_audit))
    mark('one_fullval_AP');assert not (OUT/'full_val_metrics.json').exists();metric=w.metrics(val,ids,after)
    save('full_val_metrics.json',dict(metrics=metric,images=2584,image_ids=ids,evaluations=1))
    beforematch=w.matched(val,ids,baseline);am=w.matched(val,ids,after)
    save('new_matching.json',am);save('full_val_error_analysis.json',dict(
        tiny_gained=sorted(set(am['tiny'])-set(beforematch['tiny'])),tiny_lost=sorted(set(beforematch['tiny'])-set(am['tiny'])),
        small_gained=sorted(set(am['small'])-set(beforematch['small'])),small_lost=sorted(set(beforematch['small'])-set(am['small'])),
        fp_before=len(beforematch['false_positives']),fp_after=len(am['false_positives']),evicted_instances=0))
    comp=dict(metrics=metric,baseline=bm,delta_points={k:100*(metric[k]-bm[k]) for k in metric},
              any_positive_retained=metric['APs']>g.BEST_APS,repeated_development_val=True)
    save('comparison.json',comp);finish('evaluated',1,comp)

if __name__=='__main__':main()
