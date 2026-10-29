import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from course_demo import encode_predictions
from course_batch_predict import (BatchStore, inside, output_lock, select_val_inputs,
                                  validate_rows)

class BatchTests(unittest.TestCase):
    def rows(self, image_id=1):
        mask=np.zeros((1,8,10),dtype=bool);mask[0,2:5,4:8]=True
        return encode_predictions(mask,np.array([.7]),image_id)

    def identity(self):
        return dict(model='synthetic',inputs=[
            dict(img_id=1,original_shape=[8,10]),dict(img_id=2,original_shape=[8,10])])

    def test_valid_and_empty_roundtrip(self):
        self.assertEqual(validate_rows(json.loads(json.dumps(self.rows())),1,[8,10]),1)
        self.assertEqual(validate_rows([],1,[8,10]),0)

    def test_wrong_ids_category_shape_bbox(self):
        mutations=[('image_id',2),('image_id',True),('category_id',2),('bbox',[0,0,1,1])]
        for key,value in mutations:
            rows=self.rows();rows[0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                validate_rows(rows,1,[8,10])
        rows=self.rows();rows[0]['segmentation']['size']=[9,10]
        with self.assertRaises(ValueError):validate_rows(rows,1,[8,10])

    def test_bad_scores_order_limit(self):
        for score in (float('nan'),float('inf'),-.1,1.1,True):
            rows=self.rows();rows[0]['score']=score
            with self.assertRaises(ValueError):validate_rows(rows,1,[8,10])
        rows=self.rows()*2;rows[0]=copy.deepcopy(rows[0]);rows[0]['score']=.1
        with self.assertRaises(ValueError):validate_rows(rows,1,[8,10])
        with self.assertRaises(ValueError):validate_rows(self.rows()*101,1,[8,10])

    def test_path_and_scope_guards(self):
        with tempfile.TemporaryDirectory() as td:
            for relative in ('../x','/tmp/x'):
                with self.assertRaises(ValueError):inside(td,relative)
            for ids in ([],[1,1],list(range(1,10)),[True],[0]):
                with self.assertRaises(ValueError):select_val_inputs(td,ids)
            with self.assertRaises(ValueError):select_val_inputs(Path(td)/'test',[1])

    def test_resume_coverage_empty_and_idempotence(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'run';identity=self.identity()
            with output_lock(path,False):
                s=BatchStore(path,identity,False)
                s.commit(identity['inputs'][0],self.rows())
                self.assertEqual(s.progress('partial')['status'],'partial')
            first=(path/'images/1.json').read_bytes()
            with output_lock(path,True):
                s=BatchStore(path,identity,True)
                self.assertEqual(s.initial_ids,['1'])
                s.commit(identity['inputs'][1],[])
                status=s.progress('partial')
                self.assertEqual(status['completed_ids'],[1,2])
                self.assertEqual(status['empty_ids'],[2])
                self.assertEqual(status['status'],'complete')
            self.assertEqual(first,(path/'images/1.json').read_bytes())
            with output_lock(path,True):
                self.assertEqual(BatchStore(path,identity,True).progress('complete')['instances'],1)

    def test_resume_rejects_identity_and_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'run';identity=self.identity()
            with output_lock(path,False):
                s=BatchStore(path,identity,False);s.commit(identity['inputs'][0],self.rows())
            with output_lock(path,True):
                changed=copy.deepcopy(identity);changed['model']='other'
                with self.assertRaises(ValueError):BatchStore(path,changed,True)
            file=path/'images/1.json'
            value=json.loads(file.read_text());value['rows'][0]['score']=.6
            file.write_text(json.dumps(value))
            with output_lock(path,True),self.assertRaises(ValueError):BatchStore(path,identity,True)

    def test_completed_image_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'run';identity=self.identity()
            with output_lock(path,False):
                s=BatchStore(path,identity,False);s.commit(identity['inputs'][0],self.rows())
                with self.assertRaises(ValueError):s.commit(identity['inputs'][0],self.rows())
            with self.assertRaises(FileExistsError):
                with output_lock(path,False):pass

    def test_exclusive_lock_and_resume_initialization(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'run'
            with self.assertRaises(ValueError):
                with output_lock(path,True):pass
            with output_lock(path,False):
                BatchStore(path,self.identity(),False)
                with self.assertRaises(BlockingIOError):
                    with output_lock(path,True):pass

if __name__ == '__main__':
    unittest.main(verbosity=2)
