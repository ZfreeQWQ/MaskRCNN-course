import copy
import unittest
import yaml
from course.common import configuration
from course.screen_protocol import compare_protocol


class ScreenProtocolTests(unittest.TestCase):
    def config(self,name):return yaml.safe_load(configuration(f"configs/screen/{name}.yaml").dump())

    def test_each_new_candidate_changes_exactly_one_training_factor(self):
        ref=self.config("j01_p2_queries")
        expected={"m01_p2_min_scale08":"INPUT.MIN_SCALE",
                  "m02_p2_points32768":"MODEL.MASK_FORMER.TRAIN_NUM_POINTS"}
        for name,key in expected.items():
            actual=self.config(name)
            with self.assertRaises(ValueError):compare_protocol(actual,ref)
            differences=compare_protocol(actual,ref,[key])
            self.assertEqual(set(differences),{key})
            self.assertEqual(actual["SOLVER"],ref["SOLVER"])
            self.assertEqual(actual["COURSE"]["QUERY"],ref["COURSE"]["QUERY"])

    def test_old_architecture_ablations_still_match_shared_protocol(self):
        ref=self.config("i00_control")
        for name in ("j01_p2_queries","j02_geometry_queries","k01_local_contrast","k02_reliable_contrast"):
            self.assertEqual(compare_protocol(self.config(name),ref),{})

    def test_illegal_split_seed_loss_or_lr_changes_rejected(self):
        ref=self.config("j01_p2_queries")
        for section,key,value in (("DATASETS","TRAIN",["blender_rgbd_val"]),("SOLVER","BASE_LR",.01),
                                  ("COURSE","SMALL_LOSS_WEIGHT",2)):
            actual=copy.deepcopy(ref);actual[section][key]=value
            with self.assertRaises(ValueError):compare_protocol(actual,ref,["INPUT.MIN_SCALE"])
        actual=copy.deepcopy(ref);actual["SEED"]=43
        with self.assertRaises(ValueError):compare_protocol(actual,ref,["INPUT.MIN_SCALE"])
        with self.assertRaises(ValueError):compare_protocol(ref,ref,["DATASETS.TRAIN"])

    def test_point_budget_reaches_original_matcher_and_criterion(self):
        from detectron2.modeling import build_model
        cfg=configuration("configs/screen/m02_p2_points32768.yaml",["MODEL.DEVICE","cpu"])
        model=build_model(cfg)
        self.assertEqual(model.criterion.num_points,32768)
        self.assertEqual(model.criterion.matcher.num_points,32768)
        self.assertEqual(cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,32768)


if __name__=="__main__":unittest.main()
