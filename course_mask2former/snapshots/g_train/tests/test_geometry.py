import unittest,copy
import numpy as np,torch
from geometry_signal import geometry
class Signal(unittest.TestCase):
    def setUp(self):self.K=[[100,0,31.5],[0,100,31.5],[0,0,1]]
    def test_plane(self):
        g,a=geometry(np.ones((64,64)),np.ones((64,64),bool),self.K)
        self.assertTrue(a['plane_valid']);np.testing.assert_allclose(g[2:-2,2:-2,2:5],np.broadcast_to([0,0,-1],(60,60,3)),atol=1e-7)
        self.assertLess(abs(g[...,1]).max(),1e-6);self.assertEqual(g[...,5].max(),0)
    def test_invalid(self):
        z=np.ones((64,64));z[20:40,20:40]=np.nan
        g,a=geometry(z,np.isfinite(z),self.K);self.assertTrue(np.isfinite(g).all());self.assertEqual(g[20:40,20:40].sum(),0)
        g,a=geometry(np.zeros_like(z),np.zeros_like(z,bool),self.K);self.assertEqual(g.sum(),0);self.assertFalse(a['plane_valid'])
    def test_jump(self):
        z=np.ones((64,64));z[:,32:]=2;g,a=geometry(z,np.ones_like(z,bool),self.K)
        self.assertTrue((g[:,31:33,5]>0).all());self.assertEqual(g[1:-1,31:33,2:5].sum(),0)
    def test_rng_and_replay(self):
        np.random.seed(42);before=np.random.get_state();z=np.ones((64,64));a,_=geometry(z,z>0,self.K);after=np.random.get_state()
        self.assertEqual(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(before[2:],after[2:]);b,_=geometry(z,z>0,self.K);np.testing.assert_array_equal(a,b)
    def test_slanted_normal(self):
        xx=np.indices((64,64))[1];z=1/(1+.2*(xx-31.5)/100);g,a=geometry(z,z>0,self.K)
        expected=np.array([-.2,0,-1])/np.sqrt(1.04);np.testing.assert_allclose(g[2:-2,2:-2,2:5],np.broadcast_to(expected,(60,60,3)),atol=3e-5)
    def test_camera_reject(self):
        with self.assertRaises(AssertionError):geometry(np.ones((4,4)),np.ones((4,4),bool),[[0,0,1],[0,1,1],[0,0,1]])
    def test_region_preserves_global_plane_and_neighbors(self):
        z=np.ones((64,64));z[20:35,30:45]=1.1;z[10:12,10:12]=np.nan
        full,_=geometry(z,np.isfinite(z),self.K)
        for region in [(0,0,1,1),(63,63,64,64),(0,5,1,40),(29,19,46,36),(9,9,13,13),(0,0,64,64)]:
            x0,y0,x1,y1=region;part,_=geometry(z,np.isfinite(z),self.K,region=region)
            np.testing.assert_array_equal(part,full[y0:y1,x0:x1])
    def test_region_bounds_reject(self):
        for region in [(-1,0,2,2),(0,0,65,64),(1,1,1,2)]:
            with self.assertRaises(AssertionError):geometry(np.ones((64,64)),np.ones((64,64),bool),self.K,region=region)
class Fusion(unittest.TestCase):
    def test_zero_and_active(self):
        from rgbd.course_geometry_model import GeometryFusion
        torch.manual_seed(42);m=GeometryFusion();r=tuple(torch.randn(2,c,32//2**i,32//2**i) for i,c in enumerate((256,512,1024,2048)));g=torch.randn(2,7,128,128);keep=torch.ones(2,1,1,1)
        out=m(r,g,keep);self.assertTrue(all(torch.equal(a,b) for a,b in zip(r,out)))
        sum(x.square().mean() for x in out).backward();self.assertTrue(all(p.weight.grad is not None and torch.isfinite(p.weight.grad).all() and p.weight.grad.abs().sum()>0 for p in m.projections))
        with torch.no_grad():
            for p in m.projections:p.weight.fill_(.01)
        out=m(r,g,keep);self.assertTrue(all(not torch.equal(a,b) for a,b in zip(r,out)));self.assertTrue(all(torch.equal(a,b) for a,b in zip(r,m(r,g,keep*0))))
        self.assertTrue(all(not torch.equal(a,b) for a,b in zip(out,m(r,g.flip(-1),keep))))
