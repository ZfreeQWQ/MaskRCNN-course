import io
import unittest
import torch
from torch.nn import functional as F
from detectron2.modeling import build_model
from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher
from course.common import configuration,cpu_state_dict
from course.query_screen import SeededQueryDecoder
from course.point_screen import NonlinearPointRefiner,PointEmbeddingDecoder,PointCriterion


class PointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def renderer(self,geometry=False):
        return NonlinearPointRefiner(8,16,8,16,geometry=geometry)

    def criterion(self,kind,renderer=None):
        args=(1,HungarianMatcher(num_points=32),{'loss_mask':5.,'loss_dice':5.},.1,['masks'],32,3.,.75)
        return kind(*args,**(dict(renderer=renderer) if renderer is not None else {}))

    def test_decoder_preserves_p2_masks_all_aux_and_rng(self):
        args=dict(in_channels=32,mask_classification=True,num_classes=1,hidden_dim=32,num_queries=5,
            nheads=8,dim_feedforward=64,dec_layers=2,pre_norm=False,mask_dim=16,enforce_input_project=False,
            query_mode='p2',query_channels=8)
        torch.manual_seed(4);base=SeededQueryDecoder(**args);rng=torch.get_rng_state()
        torch.manual_seed(4);new=PointEmbeddingDecoder(**args)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        new.load_state_dict(base.state_dict(),strict=True)
        x=[torch.randn(2,32,s,s) for s in (4,2,1)];low=torch.randn(2,16,8,8)
        a,b=base(x,low),new(x,low)
        for old,current in [(a,b)]+list(zip(a['aux_outputs'],b['aux_outputs'])):
            for k in ['pred_logits','pred_masks']:torch.testing.assert_close(old[k],current[k],rtol=0,atol=0)
        self.assertEqual(b['point_embedding'].shape,(2,5,16));self.assertIsNone(new._point_embedding)

    def test_subdivision_zero_residual_native_coarse_exact(self):
        m=self.renderer();coarse=torch.randn(2,5,4,4)
        fine=torch.randn(2,8,8,8);embedding=torch.randn(2,5,16)
        rng=torch.get_rng_state()
        result=m.refine(coarse,fine,embedding,points=32,chunk=2)
        torch.testing.assert_close(result,F.interpolate(coarse,size=(16,16),mode='bilinear',align_corners=False),rtol=0,atol=0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_nonzero_refinement_chunk_invariance_and_bound(self):
        m=self.renderer();torch.nn.init.normal_(m.mlp[-1].weight,std=.01)
        c=torch.randn(1,5,4,4);f=torch.randn(1,8,8,8);e=torch.randn(1,5,16)
        a=m.refine(c,f,e,points=16,chunk=1);b=m.refine(c,f,e,points=16,chunk=5)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        native=F.interpolate(c,size=(16,16),mode='bilinear',align_corners=False)
        self.assertLessEqual(float((a-native).detach().abs().max()),1.000001)

    def test_zero_loss_original_gradients_rng_and_staged_point_gradients(self):
        renderer=self.renderer();fine=torch.randn(1,8,8,8,requires_grad=True)
        embedding=torch.randn(1,2,16,requires_grad=True);coarse=torch.randn(1,2,4,4,requires_grad=True)
        target=[dict(masks=(torch.rand(2,16,16)>.7).float())];ids=[(torch.arange(2),torch.arange(2))]
        torch.manual_seed(8);a=self.criterion(SetCriterion).loss_masks(dict(pred_masks=coarse),target,ids,2)
        rng=torch.get_rng_state();a_grad=torch.autograd.grad(sum(a.values()),coarse)[0]
        torch.manual_seed(8)
        criterion=self.criterion(PointCriterion,renderer)
        out=dict(pred_masks=coarse,point_fine=fine,point_embedding=embedding)
        b=criterion.loss_masks(out,target,ids,2)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
        sum(b.values()).backward();torch.testing.assert_close(a_grad,coarse.grad,rtol=0,atol=0)
        self.assertGreater(float(renderer.mlp[-1].weight.grad.abs().sum()),0)
        torch.nn.init.normal_(renderer.mlp[-1].weight,std=.01)
        renderer.zero_grad();coarse.grad=None
        sum(criterion.loss_masks(out,target,ids,2).values()).backward()
        for p in [renderer.query.weight,renderer.mlp[0].weight]:
            self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0)
        self.assertGreater(float(fine.grad.abs().sum()),0)

    def test_render_groups_images_without_cross_image_leakage(self):
        m=self.renderer();torch.nn.init.normal_(m.mlp[-1].weight,std=.1)
        f=torch.randn(2,8,8,8);e=torch.randn(3,16);c=torch.randn(3,32);coords=torch.rand(3,32,2)
        image_ids=torch.tensor([0,1,0])
        a=m.render(f,e,c,coords,image_ids)
        f[1]*=50;b=m.render(f,e,c,coords,image_ids)
        torch.testing.assert_close(a[[0,2]],b[[0,2]],rtol=0,atol=0)
        self.assertFalse(torch.allclose(a[1],b[1]))
        with self.assertRaises(ValueError):m.render(f,e,c,coords,torch.tensor([0,2,0]))

    def test_geometry_off_no_effect_on_common_parameters_and_invalid_pixels(self):
        torch.manual_seed(42);a=self.renderer()
        torch.manual_seed(42);b=self.renderer(True)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        rgb=torch.randn(2,8,8,8);geometry=torch.zeros(2,7,8,8)
        torch.testing.assert_close(a.fine_features(rgb),b.fine_features(rgb,geometry),rtol=0,atol=0)
        with self.assertRaises(ValueError):b.fine_features(rgb)

    def test_empty_targets_and_auxiliary_path(self):
        m=self.renderer();criterion=self.criterion(PointCriterion,m)
        c=torch.randn(1,2,4,4,requires_grad=True);f=torch.randn(1,8,8,8,requires_grad=True)
        loss=criterion.loss_masks(dict(pred_masks=c,point_fine=f,point_embedding=torch.randn(1,2,16)),
            [dict(masks=torch.zeros(0,16,16))],[(torch.zeros(0,dtype=torch.long),torch.zeros(0,dtype=torch.long))],1)
        sum(loss.values()).backward();self.assertEqual(float(sum(loss.values()).detach()),0)
        target=[dict(masks=torch.zeros(2,16,16))];ids=[(torch.arange(2),torch.arange(2))]
        torch.manual_seed(10);a=self.criterion(SetCriterion).loss_masks(dict(pred_masks=c),target,ids,2)
        torch.manual_seed(10);b=criterion.loss_masks(dict(pred_masks=c),target,ids,2)
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)

    def test_full_shared_initialization_strict_state_metadata_and_single_owner(self):
        torch.manual_seed(42)
        base=build_model(configuration('configs/screen/j01_p2_queries.yaml',['MODEL.DEVICE','cpu']))
        rng=torch.get_rng_state();original=base.state_dict();states=[]
        for n in ['o01_p2_point_rgbd','o02_p2_point_geometry']:
            torch.manual_seed(42);cfg=configuration(f'configs/screen/{n}.yaml',['MODEL.DEVICE','cpu'])
            model=build_model(cfg)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            state=model.state_dict();self.assertTrue(all(torch.equal(v,state[k]) for k,v in original.items()))
            self.assertTrue(all(k.startswith('sem_seg_head.point_refiner.') for k in set(state)-set(original)))
            self.assertIs(model.criterion._renderer_ref(),model.sem_seg_head.point_refiner)
            stream=io.BytesIO();torch.save(dict(model=cpu_state_dict(model)),stream);stream.seek(0)
            clone=build_model(cfg);clone.load_state_dict(torch.load(stream,weights_only=False)['model'],strict=True)
            self.assertIs(clone.criterion._renderer_ref(),clone.sem_seg_head.point_refiner)
            self.assertEqual(clone.state_dict()._metadata['sem_seg_head']['version'],2)
            states.append(model.sem_seg_head.point_refiner.state_dict())
        self.assertTrue(all(torch.equal(v,states[1][k]) for k,v in states[0].items()))

    def test_point_lr_only_new_module_and_backbone_dropout_rng(self):
        from train_screen import ScreenTrainer
        torch.manual_seed(42)
        base=build_model(configuration('configs/screen/j01_p2_queries.yaml',['MODEL.DEVICE','cpu'])).backbone.train()
        torch.manual_seed(42);cfg=configuration('configs/screen/o02_p2_point_geometry.yaml',['MODEL.DEVICE','cpu'])
        model=build_model(cfg);backbone=model.backbone.train();backbone.load_state_dict(base.state_dict(),strict=True)
        x=torch.randn(2,11,64,64)
        torch.manual_seed(5);a=base(x);rng=torch.get_rng_state()
        torch.manual_seed(5);b=backbone(x)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
        expected=F.interpolate(x[:,4:]*backbone.geometry_fusion.last_keep,size=(32,32),mode='nearest')
        torch.testing.assert_close(b['point_geometry'],expected,rtol=0,atol=0)
        self.assertIsNone(backbone.stem.half_feature)
        opt=ScreenTrainer.build_optimizer(cfg,model);names={id(p):n for n,p in model.named_parameters()}
        for group in opt.param_groups:
            for p in group['params']:
                name=names[id(p)]
                if '.point_refiner.' in name or '.query_initializer.' in name:expected=1e-4
                elif name.startswith('backbone.') and not name.startswith('backbone.geometry_fusion.'):expected=1e-6
                else:expected=1e-5
                self.assertAlmostEqual(group['lr'],expected)


if __name__=='__main__':unittest.main()
