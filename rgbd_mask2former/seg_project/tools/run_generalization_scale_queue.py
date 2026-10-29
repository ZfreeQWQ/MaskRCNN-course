#!/usr/bin/env python
"""Insert a scale/coverage experiment after matched 6ep, before 12ep training."""
import fcntl
from pathlib import Path
from algorithm_queue_worker import now
from run_algorithm_queue import read
from run_generalization_long_queue import Queue as LongQueue, ROOT, OUT


class Queue(LongQueue):
    def scale_experiment(self):
        if self.state.get('multiview_completed'):
            return
        if not read(OUT/'multiview_cpu_tests.json')['valid']:
            raise RuntimeError('Multiview CPU/real mapper checks are required')
        eligible=[v for v in self.state['candidates'].values() if v['epochs']==6 and
                  v['AP']>=self.state['baseline']['AP']-.3]
        selected=max(eligible,key=lambda v:(v['APs'],v['AP']))
        root=OUT/'multiview_20261028'
        state=self.state.setdefault('multiview',{})
        state.update(dict(source=selected['output'],
            source_APs=selected['APs'],weights_sha256=selected['weights_sha256'],
            objective='large small-object gain; reported val-test gap is not assumed constant',
            subset_rule='64 SHA256-selected val scenes; held-out remaining scenes',
            pipeline='CPU tests -> four-GPU reference smoke -> pilot -> full+holdout -> optional test CSV',
            test_annotations_used=False))
        self.save()
        def run(phase,variant='all',confirmation=None):
            state.update(status='running',phase=phase,output=str(root/phase));self.save()
            command=[str(ROOT/'run.sh'),'python','-u',str(ROOT/'tools/eval_multiview.py'),
                '--source',state['source'],'--output',str(root/phase),'--phase',phase,'--variant',variant]
            if confirmation:
                command.extend(['--confirmation',str(confirmation)])
            self.job('multiview_'+phase,command,timeout=10800)
            result=read(root/phase/'summary.json')
            if not result['valid']:
                raise RuntimeError('Invalid multiview result')
            state[phase]=result;self.save()
            return result
        try:
            run('smoke')
            pilot=run('pilot')
            acceptable={k:v for k,v in pilot['results'].items() if v['delta_AP']>=-.3}
            winner=max(acceptable,key=lambda k:acceptable[k]['delta_APs']) if acceptable else None
            if winner and acceptable[winner]['delta_APs']>=1.:
                full=run('full',winner)
                if full['full_confirmed']:
                    run('test',winner,root/'full/summary.json')
                    state['conclusion']='full_and_unseen_val_scenes_confirmed; new CSV ready for Kaggle'
                else:
                    state['conclusion']='pilot_gain_did_not_confirm_on_full_and_heldout_scenes'
            else:
                state['conclusion']='no_at_least_1_point_pilot_gain_with_AP_guard; no_test_export'
            state.update(status='completed',completed_utc=now())
        except RuntimeError as error:
            # A new inference hypothesis failing must not discard or halt valid
            # optimizer-preserving control/algorithm training. Never retry in place.
            state.update(status='failed',error=repr(error),completed_utc=now())
        self.state['multiview_completed']=True
        self.save()

    def continuation(self,name,parent,epochs):
        if epochs==12:
            self.scale_experiment()
        return super().continuation(name,parent,epochs)


if __name__=='__main__':
    with (OUT/'manager.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        queue=Queue()
        try:
            queue.execute()
        except BaseException as error:
            queue.state.update(status='failed',error=repr(error));queue.save();raise
