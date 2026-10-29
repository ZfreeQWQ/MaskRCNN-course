import unittest
import numpy as np
from g_system_worker import geometry_crop,sync_crop,gpu_source,a
class System(unittest.TestCase):
 def test_geometry_crops(self):
  g=np.arange(1024*1024*7,dtype=np.float32).reshape(1024,1024,7)
  for size,origins in ((640,[(0,0),(384,0),(0,384),(384,384)]),(512,[(0,0),(512,0),(0,512),(512,512)])):
   for x,y in origins:
    out=geometry_crop(g,(x,y),size);np.testing.assert_array_equal(out,g[y:y+size,x:x+size]);self.assertFalse(np.shares_memory(out,g))
  np.testing.assert_array_equal(geometry_crop(g,None,1024),g)
 def test_new_global_binding_required(self):self.assertIn("compare_global(actual,b['image_id'])",gpu_source())
 def test_frozen_functions(self):self.assertEqual(set(a.numeric_functions()),{'m2f_flags','head_flags','new_views','actual_full'})

 def test_original_crop_planes_and_geometry_align(self):
  import m2f_tiles as tiles
  g=np.arange(1024*1024*7,dtype=np.float32).reshape(1024,1024,7)
  d=np.arange(1024*1024,dtype=np.float32).reshape(1024,1024)
  raw=dict(img=np.repeat((d%255)[...,None],3,axis=-1).astype(np.uint8),rgbd_depth=d,rgbd_raw_valid=d>100,rgbd_label_valid=d>200,geometry=g,ori_shape=(1024,1024),img_shape=(1024,1024))
  for fn,size,origins in [(tiles.crop_loaded,640,tiles.WINDOWS),(a.old.crop512,512,a.old.WINDOWS)]:
   for xy in origins:
    before=fn(raw,xy);after=sync_crop(fn,raw,xy,size)
    for k in ('img','rgbd_depth','rgbd_raw_valid','rgbd_label_valid'):np.testing.assert_array_equal(before[k],after[k])
    self.assertEqual(before['ori_shape'],after['ori_shape']);x,y=xy
    np.testing.assert_array_equal(after['geometry'],g[y:y+size,x:x+size])

 def test_supervisor_protocol_not_shadowed_by_process_result(self):
  import ast,g_system_launcher as launch
  source=launch.transformed();tree=ast.parse(source)
  execute=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='execute')
  self.assertTrue(any(isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='gproto' and n.attr=='TRAIN' for n in ast.walk(execute)))
  self.assertFalse(any(isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='p' and n.attr in ('TRAIN','UUID','UUID_LOCK') for n in ast.walk(tree)))
  self.assertFalse(any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id=='gproto' for n in ast.walk(tree)))
  self.assertIn("if gpu:assert h['DeviceRequests'][0]['DeviceIDs']==[str(selected)]",source)
  self.assertIn("if now-stamp>300",source)
  compile(source,'<G supervisor scope regression>','exec')

 def test_generated_GPU_loop_protocol_scope(self):
  import ast,g_system_launcher as launch
  from g_system_worker import gpu_source
  for source in (gpu_source(),launch.transformed()):
   tree=ast.parse(source)
   self.assertFalse(any(isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='p' and n.attr.isupper() for n in ast.walk(tree)))
   self.assertFalse(any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id=='gproto' for n in ast.walk(tree)))
   compile(source,'<both emitted functions protocol scope>','exec')
  self.assertIn("gproto.OLD_WEIGHT",gpu_source())
  self.assertIn("gproto.WEIGHT",gpu_source())
  self.assertIn("p.PREVIOUS_GPU_SECONDS",__import__('inspect').getsource(launch.lease))
