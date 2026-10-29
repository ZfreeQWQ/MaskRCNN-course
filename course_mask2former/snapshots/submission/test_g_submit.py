import copy,inspect,json,unittest
import numpy as np
from pycocotools import mask as mu
import g_submit_codec as c
import g_submit_worker as w
import g_submit_spec as p
class SubmissionTests(unittest.TestCase):
 def row(self,area=1,score=.7):
  a=np.zeros((1024,1024),np.uint8);a.flat[:area]=1;r=mu.encode(np.asfortranarray(a));r['counts']=r['counts'].decode();return dict(image_id=7,category_id=1,score=score,segmentation=r,bbox=mu.toBbox(r).tolist())
 def test_drop_only_empty(self):
  a,b=self.row(2,.2),self.row(4,.8);out,n=c.filter_rows([a,self.row(0,0),b],7);self.assertEqual(n,1);self.assertEqual(out,[{k:r[k] for k in ('image_id','category_id','score','segmentation')} for r in (a,b)])
 def test_all_empty(self):self.assertEqual(c.filter_rows([self.row(0)],7),([],1))
 def test_invalid_score(self):
  for score in (float('nan'),float('inf'),-1,1.1):
   with self.assertRaises(AssertionError):c.filter_rows([self.row(score=score)],7)
 def test_invalid_id(self):
  with self.assertRaises(AssertionError):c.filter_rows([self.row()],8)
 def test_wrong_bbox(self):
  r=self.row();r['bbox']=[0,0,0,0]
  with self.assertRaises(AssertionError):c.filter_rows([r],7)
 def test_wrong_size(self):
  r=self.row();r['segmentation']['size']=[512,512]
  with self.assertRaises(AssertionError):c.filter_rows([r],7)
 def test_max100(self):
  with self.assertRaises(AssertionError):c.filter_rows([self.row()]*101,7)
 def test_ids(self):
  with self.assertRaises(AssertionError):c.ids_checked([1,1])
 def test_safe_input_scope(self):
  docs=[dict(id=i,split='test',width=1024,height=1024) for i in range(2112)];self.assertEqual(len(w.safe_images(docs)),2112)
  docs[0]['annotations']=[]
  with self.assertRaises(AssertionError):w.safe_images(docs)
 def test_geometry_index_binding(self):
  req=dict(depth_scale_m=.0001,depth_invalid_code=0,depth_file='/workspace/test/depth.png',img_path='/workspace/test/rgb.png')
  manifest=[dict(image_id=i,request=req) for i in range(2112)];K=[[1,0,.5],[0,1,.5],[0,0,1]];ks={i:K for i in range(2112)}
  refs=[dict(base=dict(image_id=3000+i)) for i in range(8)];orig=dict(images={str(3000+i):dict(K=K,split='val') for i in range(8)})
  out=w.make_geometry_index(manifest,ks,orig,refs);self.assertEqual(len(out['images']),2120);self.assertEqual(out['images']['0']['K'],K);self.assertEqual(out['images']['3000'],orig['images']['3000'])
  ks[1]=[[float('nan'),0,0],[0,1,0],[0,0,1]]
  with self.assertRaises(AssertionError):w.make_geometry_index(manifest,ks,orig,refs)
 def test_fixed_lease_and_gpu(self):
  self.assertEqual(p.GPU_MAX,14400);self.assertEqual(p.DEADLINE-p.START,21600);self.assertIn('4305f6',p.UUID)
 def test_runtime_only_changes_input_index(self):
  src=inspect.getsource(w.g.geometry_runtime);self.assertEqual(src.count("pipe[1]['scale']=(2559,1536)"),1);self.assertIn('submission_index',inspect.getsource(w.geometry_runtime))
if __name__=='__main__':unittest.main()
