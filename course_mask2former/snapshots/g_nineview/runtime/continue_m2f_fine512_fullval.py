"""Continue immutable v1 predictions with a new independently bounded lease."""
import ast, inspect, os, shutil, time, unittest
from pathlib import Path
import run_m2f_fine512_fullval as v

START=1792533242
PREP_END,GPU_END,ANALYSIS_END,DEADLINE=(START+n for n in (900,9000,9900,10800))
NAME='course_m2f_fine512_fullval_20261021_v2'
OWNER='be2256b59e944bff89b67c8fc0190759'
ROOT=Path('/workspace');OUT=ROOT/'outputs'/NAME
PARENT=ROOT/'outputs/course_m2f_fine512_fullval_20261021_v1'
read,save,sha=v.read,v.save,v.sha
for k in ('START','PREP_END','GPU_END','ANALYSIS_END','DEADLINE','NAME','OWNER','OUT'):
    setattr(v,k,globals()[k])
v.old.START,v.old.DEADLINE=START,DEADLINE

def execution_order(ids,fixed):
    reused=ids[:32];missing=[i for i in fixed if i not in reused]
    remaining=[i for i in ids if i not in reused and i not in missing]
    assert len(missing)==4 and len(remaining)==2548
    assert set(reused+missing+remaining)==set(ids) and len(set(ids))==2584
    return reused,missing,remaining

# Only orchestration changes. Nested numerical production functions are checked
# for exact AST identity before compilation and again in CPU acceptance.
LOOP='''    manifest=read(OUT/'prepared_manifest.json');ids=manifest['ids'];index={x['image_id']:x for x in manifest['files']}
    reused,missing,remaining=continuation_order(ids,w.base.FIXED8)
    captured=[];capture_seconds=[]
    def capture_one(i):
        t=time.time();assert t<end;p=OUT/'base_cache'/f'{i}.json';assert sha(p)==index[i]['sha256'];cache=read(p);request=cache['request']
        assert not (OUT/'captures'/f'{i}.json').exists() and not (OUT/'predictions'/f'{i}.json').exists()
        ih=dict(image_id=i,rgb_sha256=sha(request['img_path']),depth_sha256=sha(request['depth_file']));assert ih==cache['input_hashes']
        new,origins=new_views(request,i);rgbd=ref.load_rgbd(request)
        final,activity=combine(cache['current'],cache['merged'],cache['jobs'],cache['probabilities'],new,rgbd)
        assert tiny.merge(cache['current'],cache['merged'],cache['jobs'],cache['probabilities'])==(cache['before'],cache['before_activity'])
        save(OUT/'captures'/f'{i}.json',dict(image_id=i,new_raw=new,new_origins=origins,rows=final,activity=activity,input_hashes=ih))
        save(OUT/'predictions'/f'{i}.json',final);captured.append(i);capture_seconds.append(time.time()-t)
        if len(captured)%16==0:mark('new_views_val',images=32+len(captured),new_images=len(captured),expected=2584,eval_calls=calls['eval'],train_calls=calls['train'])
        if len(captured)==32:
            projected=sum(capture_seconds)/32*(2552-32)*1.15+240
            save(OUT/'throughput_gate.json',dict(status='passed' if time.time()+projected<end else 'failed',new_images=32,mean_seconds=sum(capture_seconds)/32,projected_remaining_seconds=projected,end=end,unix=time.time(),parity_time_excluded_from_capture_mean=True))
            assert time.time()+projected<end,'Independent budget insufficient; no protocol reduction'
    with torch.inference_mode():
        for i in missing:capture_one(i)
        parity=[]
        for i in w.base.FIXED8:
            cache=read(OUT/'base_cache'/f'{i}.json');saved=read(OUT/'captures'/f'{i}.json');actual,ms=actual_full(cache)
            for key in ('global_rows','current','merged','jobs','probabilities'):assert actual[key]==cache[key],(i,key)
            for key in ('new_raw','new_origins','rows','activity'):assert actual[key]==saved[key],(i,key)
            parity.append(dict(image_id=i,old_outputs_exact=True,new_raw_exact=True,probabilities_exact=True,final_instances_exact=True))
            mark('actual_parity',images=len(parity),expected=8)
        save(OUT/'parity.json',parity)
        assert calls==dict(eval=4*4+8*9,train=0)
        for i in remaining:capture_one(i)
        assert len(captured)==len(set(captured))==2552 and not set(captured)&set(reused)
        assert calls==dict(eval=2552*4+8*9,train=0)
        files=[];predictions=[];inputs=[]
        for i in ids:
            p=OUT/'base_cache'/f'{i}.json';assert sha(p)==index[i]['sha256'];cache=read(p)
            capture=OUT/'captures'/f'{i}.json';output=OUT/'predictions'/f'{i}.json';c=read(capture);rows=read(output)
            assert c['image_id']==i and c['rows']==rows and c['input_hashes']==cache['input_hashes']
            assert len(rows)<=100 and all(r['image_id']==i and r['segmentation']['size']==[1024,1024] for r in rows)
            predictions.extend(rows);inputs.append(c['input_hashes']);files.append(dict(image_id=i,capture_sha256=sha(capture),prediction_sha256=sha(output),reused=i in reused))
        save(OUT/'predictions.segm.json',predictions);save(OUT/'val_input_hashes.json',inputs)
        save(OUT/'predictions_frozen.json',dict(images=2584,image_ids=ids,files=files,prediction_sha256=sha(OUT/'predictions.segm.json'),unix=time.time(),reused_images=32,new_images=2552))
        del predictions
        cache=read(OUT/'base_cache'/f'{w.base.FIXED8[0]}.json');expected=read(OUT/'captures'/f'{w.base.FIXED8[0]}.json');times=[];peaks=[];reserved=[]
        for n in range(25):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();actual,ms=actual_full(cache)
            assert actual['rows']==expected['rows'] and actual['activity']==expected['activity']
            if n>=5:times.append(ms);peaks.append(torch.cuda.max_memory_allocated()/2**20);reserved.append(torch.cuda.max_memory_reserved()/2**20)
            mark('full_timing',completed=n+1,expected=25)
'''

def compiled_gpu():
    original=inspect.getsource(v.gpu)
    first=original.index("    manifest=read(OUT/'prepared_manifest.json')")
    last=original.index('    handle.remove()')
    modified=original[:first]+LOOP+original[last:]
    assert modified.count('eval=10633')==1
    modified=modified.replace('eval=10633','eval=10505')
    def numerical(text):
        fn=ast.parse(text).body[0]
        return {n.name:ast.dump(n,include_attributes=False) for n in fn.body if isinstance(n,ast.FunctionDef) and n.name!='capture_one'}
    assert numerical(original)==numerical(modified)
    assert set(numerical(original))=={'head','hook','m2f_flags','head_flags','new_views','combine','actual_full'}
    ns=dict(v.__dict__,continuation_order=execution_order)
    exec(compile(modified,'<frozen-gpu-with-resume-orchestration>','exec'),ns)
    return ns['gpu'],modified

class ResumeContracts(unittest.TestCase):
    def test_budget(self):
        self.assertEqual((PREP_END-START,GPU_END-START,ANALYSIS_END-START,DEADLINE-START),(900,9000,9900,10800))
    def test_order_and_no_recapture(self):
        ids=[s*1000+n for s in range(201960,202283) for n in range(1,9)]
        a,b,c=execution_order(ids,[s*1000+1 for s in range(201960,201968)])
        self.assertEqual((len(a),len(b),len(c)),(32,4,2548));self.assertFalse(set(a)&set(b+c))
    def test_calls(self):self.assertEqual(2552*4+33*9,10505);self.assertEqual(10505+128,10633)
    def test_numerical_identity(self):self.assertTrue(callable(compiled_gpu()[0]))
    def test_official_order_restored(self):self.assertIn('for i in ids:',LOOP)
    def test_parity_before_bulk(self):self.assertLess(LOOP.index("save(OUT/'parity.json'"),LOOP.index('for i in remaining:'))
    def test_no_new_threshold(self):self.assertNotIn('probability=',LOOP);self.assertIn('*1.15+240',LOOP)

def prep():
    import m2f_tiny_verifier as tiny
    v.cutoff(PREP_END);assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not Path('/dev/nvidiactl').exists()
    suite=unittest.TestSuite()
    for case,excluded in ((v.old.Contracts,{'test_budget'}),(v.Contracts,{'test_budget_calls'}),(ResumeContracts,set())):
        for n in unittest.defaultTestLoader.getTestCaseNames(case):
            if n not in excluded:suite.addTest(case(n))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName('test_m2f_tiny_verifier'))
    r=unittest.TextTestRunner(verbosity=2).run(suite);assert r.wasSuccessful() and not r.skipped
    assert read(PARENT/'status.json')['status']=='failed' and read(PARENT/'throughput_gate.json')['images']==32
    assert read(PARENT/'cpu_acceptance.json')['status']=='passed' and not (PARENT/'full_val_metrics.json').exists()
    obs=OUT/'seed_evidence/remote_closure_observation.json';assert sha(obs)=='ccb660799d9871bbad89795a7e00f486d180f9d7ad49b77351f21ac8cc9e8764'
    observation=read(obs);assert not observation['supervisor_alive'] and not observation['owner_containers']
    for n,h in read(PARENT/'plan.json')['source_sha256'].items():assert sha(PARENT/'source'/n)==sha(OUT/'source'/n)==h,n
    for n,h in observation['archived_file_sha256'].items():assert sha(PARENT/n)==h,n
    binding=read(PARENT/'binding.json');assert all(sha(ROOT/n)==h for n,h in binding['input_sha256'].items())
    mf=read(PARENT/'prepared_manifest.json');assert mf['all_2584_old_outputs_exact'] and not mf['gt_to_model']
    reused,missing,remaining=execution_order(mf['ids'],v.w.base.FIXED8)
    assert {p.stem for p in (PARENT/'captures').glob('*.json')}=={str(i) for i in reused}
    assert {p.stem for p in (PARENT/'predictions').glob('*.json')}=={str(i) for i in reused}
    (OUT/'base_cache').symlink_to('../'+PARENT.name+'/base_cache',target_is_directory=True)
    for name in ('prepared_manifest.json','binding.json'):shutil.copyfile(PARENT/name,OUT/name)
    for name in ('captures','predictions'):(OUT/name).mkdir()
    for item in mf['files']:
        v.cutoff(PREP_END);i=item['image_id'];p=PARENT/'base_cache'/f'{i}.json';assert sha(p)==item['sha256']
        if i in reused:
            c=read(PARENT/'captures'/f'{i}.json');b=read(p);assert c['input_hashes']==b['input_hashes']
            assert c['rows']==read(PARENT/'predictions'/f'{i}.json')
            jobs=c['activity']['candidates'];assert v.jobs_without_p(jobs[:len(b['jobs'])])==b['jobs']
            assert [j['probability'] for j in jobs[:len(b['jobs'])]]==b['probabilities']
            actual=tiny.merge(b['current'],b['merged']+c['new_raw'],v.jobs_without_p(jobs),[j['probability'] for j in jobs])
            assert actual==(c['rows'],{k:x for k,x in c['activity'].items() if k!='candidates'})
            for name in ('captures','predictions'):shutil.copyfile(PARENT/name/f'{i}.json',OUT/name/f'{i}.json')
    save(OUT/'reuse_manifest.json',dict(parent=str(PARENT),reused_ids=reused,missing_fixed=missing,new_ids=missing+remaining,parent_observation_sha256=sha(obs),inferred_previous_eval_calls=128,previous_final_hook_report_exists=False))
    fn,code=compiled_gpu();save(OUT/'gpu_orchestration.json',dict(source=code,numerical_nested_ast_exact=True))
    save(OUT/'cpu_acceptance.json',dict(status='passed',tests=r.testsRun,skipped=0,reused_images=32,base_cache_sha_count=2584,old_ap_recomputed=False,unix=time.time()))

def gpu():
    assert read(OUT/'cpu_acceptance.json')['status']=='passed'
    compiled_gpu()[0]()
    r=read(OUT/'gpu_report.json');assert r['calls']==dict(eval=10505,train=0)
    r.update(reused_capture_images=32,old_inferred_eval_calls=128,combined_protocol_eval_calls=10633)
    save(OUT/'gpu_report.json',r)

def finish():v.finish()

if __name__=='__main__':
    import sys
    {'prep':prep,'gpu':gpu,'finish':finish}[sys.argv[1]]()
