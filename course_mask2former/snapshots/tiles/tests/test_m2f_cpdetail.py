import unittest
from types import SimpleNamespace
from unittest.mock import patch
import m2f_cpdetail as arm
import run_m2f_cpdetail as launch

class Contract(unittest.TestCase):
    def test_only_one_job(self):
        self.assertEqual(arm.job()['id'],'copy_paste_detail_long')
        self.assertEqual(sum(arm.job()['flags'].values()),4)
    def test_reject_other_treatment(self):
        a=SimpleNamespace(**arm.FLAGS,course_depth_reliability=False,course_high_resolution=False)
        arm.check_flags(a)
        for k in ('course_depth_reliability','course_high_resolution'):
            setattr(a,k,True)
            with self.assertRaises(ValueError):arm.check_flags(a)
            setattr(a,k,False)
    def test_both_required(self):
        a=SimpleNamespace(**arm.FLAGS,course_depth_reliability=False,course_high_resolution=False)
        for k in arm.FLAGS:
            setattr(a,k,False)
            with self.assertRaises(ValueError):arm.check_flags(a)
            setattr(a,k,True)
    def test_model_delta_only(self):
        original=dict(type='RGBDMask2Former',keep=dict(n=100))
        new=arm.expected_model(original)
        self.assertEqual(new['keep'],original['keep']);self.assertNotIn('course_detail',original)
        self.assertEqual(set(new)-set(original),{'course_detail','course_depth_reliability'})
    def test_fresh_training(self):
        command=launch.train_args(launch.ROOT/'outputs/example',False)
        self.assertIn('/workspace/'+launch.queue.CHECKPOINT,command)
        self.assertIn('30000',command);self.assertNotIn('--resume',command)
        self.assertEqual(command.count('--course-detail'),1);self.assertIn('--course-copy-paste',command)
    def test_smoke20(self):
        command=launch.train_args(launch.ROOT/'outputs/example',True)
        self.assertEqual(command[command.index('--iters')+1],'20');self.assertIn('--functional-smoke',command)
    def test_budget_stop(self):
        s=launch.Single(launch.ROOT/'outputs/example',dict(max_seconds=43200,deadline_unix=1))
        with self.assertRaises(TimeoutError):s.check_stop()
    def test_container_isolation(self):
        _,c=launch.docker(launch.ROOT/'outputs/example',dict(owner_token='abc',image_id='image'),'test',['python','test.py'],True)
        self.assertIn('device=0',c);self.assertIn('--read-only',c);self.assertIn('none',c)
        self.assertIn('ai-course-directions-owner=abc',c)
        self.assertEqual(c.count('--volume'),3)

if __name__=='__main__':unittest.main()
