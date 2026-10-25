import copy
import unittest
from course.confirmation import select_winner,validate_panels


class ConfirmationTests(unittest.TestCase):
    def test_selection_uses_final_and_consistent_steps_not_lucky_peak(self):
        control=[dict(additional_training_step=k,mask_aps=.45) for k in (0,1000,2000,3000)]
        jobs={n:dict(status="completed",screen_metrics=[dict(additional_training_step=k,mask_aps=v)
              for k,v in zip((0,1000,2000,3000),values)]) for n,values in
              {"peak_only":[.45,.44,.7,.44],"steady":[.45,.46,.47,.48],"weaker":[.45,.46,.47,.475]}.items()}
        self.assertEqual(select_winner(jobs,control),"steady")
        jobs["steady"]["status"]="running"
        with self.assertRaises(ValueError):select_winner(jobs,control)

    def test_no_positive_final_no_confirmation(self):
        control=[dict(additional_training_step=k,mask_aps=.45) for k in (1000,2000,3000)]
        jobs={"negative":dict(status="completed",screen_metrics=[dict(additional_training_step=k,mask_aps=v)
              for k,v in zip((1000,2000,3000),(.46,.49,.44))])}
        self.assertIsNone(select_winner(jobs,control))

    def test_disjoint_scene_panel_guard(self):
        a=dict(split="val",scenes=list(range(100)),image_ids=list(range(200)))
        b=dict(split="val",scenes=list(range(100,200)),image_ids=list(range(200,400)))
        validate_panels(a,b)
        c=copy.deepcopy(b);c["scenes"][0]=0
        with self.assertRaises(ValueError):validate_panels(a,c)
        c=copy.deepcopy(b);c["image_ids"][0]=0
        with self.assertRaises(ValueError):validate_panels(a,c)
        c=copy.deepcopy(b);c["split"]="test"
        with self.assertRaises(ValueError):validate_panels(a,c)


if __name__=="__main__":unittest.main()
