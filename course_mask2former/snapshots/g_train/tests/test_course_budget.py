"""One bounded long-baseline run, with no hidden algorithm/inference changes."""
import unittest
from run_course_workflow import variants, formal_protocol, validate_gpus
from run_course_experiment import parser, build_config
from launch_nyu_sweep import docker_command


class CourseBudgetTests(unittest.TestCase):
    def test_only_one_unchanged_gated_variant(self):
        self.assertEqual(variants('budget_probe'), [('gated_depth_long',dict(multiscale_depth=True,reliability_gate=True))])
        self.assertEqual(variants('budget_probe')[0][1], variants('baseline')[1][1])

    def test_predeclared_budget_and_legacy_pairs(self):
        p = formal_protocol('budget_probe')
        self.assertEqual((p['stage'],p['iters'],p['val_interval'],p['keep_checkpoints']),
                         ('budget_diagnostic',30000,5000,6))
        self.assertEqual(p['max_job_seconds'],43200)
        for pair in ('baseline','small_supervision','scale_crop'):
            p = formal_protocol(pair)
            self.assertEqual((len(variants(pair)),p['iters'],p['val_interval'],p['keep_checkpoints']), (2,10000,5000,2))
        with self.assertRaises(ValueError): formal_protocol('unknown')

    def test_gpu_count_and_busy_checks_prevent_silent_zip_truncation(self):
        idle = {0:dict(memory_used_mib=0,utilization=0),1:dict(memory_used_mib=0,utilization=0)}
        validate_gpus('budget_probe',[0],idle)
        validate_gpus('baseline',[0,1],idle)
        for pair,gpus in [('budget_probe',[0,1]),('baseline',[0]),('baseline',[0,0]),('budget_probe',[])]:
            with self.assertRaises(ValueError): validate_gpus(pair,gpus,idle)
        with self.assertRaises(RuntimeError): validate_gpus('budget_probe',[2],idle)
        busy = dict(idle);busy[0] = dict(memory_used_mib=1024,utilization=0)
        with self.assertRaises(RuntimeError): validate_gpus('budget_probe',[0],busy)
        validate_gpus('budget_probe',[1],busy)  # Unselected devices are untouched.

    def config(self,iters=10000,keep=2):
        args=parser().parse_args(['train','--output','/tmp/course_budget_config_only',
            '--multiscale-depth','--reliability-gate','--iters',str(iters),
            '--val-interval','5000','--keep-checkpoints',str(keep)])
        return build_config(args,[dict(id=1,name='component',supercategory='component')])

    def test_long_recipe_only_budget_schedule_and_checkpoint_retention_change(self):
        base = self.config().to_dict()
        long = self.config(30000,6).to_dict()
        self.assertEqual(long['param_scheduler'][0]['milestones'],[18000,24000])
        self.assertEqual(long['param_scheduler'][0]['gamma'],.1)
        self.assertEqual(long['train_cfg']['val_interval'],5000)
        self.assertFalse(long['resume'])
        long['train_cfg']['max_iters']=base['train_cfg']['max_iters']
        long['param_scheduler']=base['param_scheduler']
        long['default_hooks']['checkpoint']['max_keep_ckpts']=2
        self.assertEqual(long,base)

    def test_checkpoint_retention_must_be_positive(self):
        with self.assertRaises(ValueError): self.config(30000,0)

    def test_command_transmits_budget_and_only_declared_switches(self):
        plan=dict(workspace='/mnt/nfs/fenghaibin/AI_Class',container_output_root='/workspace/outputs/example/budget_diagnostic',
            container_name_prefix='test-course',image='test-image',runner_script='run_course_experiment.py',
            recipe=dict(iters=30000,val_interval=5000,keep_checkpoints=6,schedule='decay'))
        identity,flags=variants('budget_probe')[0]
        _,cmd=docker_command(plan,dict(id=identity,seed=20261013,output_rel='runs/'+identity,flags=flags),0)
        for key,value in [('--iters','30000'),('--val-interval','5000'),('--keep-checkpoints','6')]:
            self.assertEqual(cmd[cmd.index(key)+1],value)
        self.assertIn('--multiscale-depth',cmd);self.assertIn('--reliability-gate',cmd)
        self.assertNotIn('--small-object-loss',cmd);self.assertNotIn('--course-large-scale-jitter',cmd)


if __name__ == '__main__':
    unittest.main(verbosity=2)
