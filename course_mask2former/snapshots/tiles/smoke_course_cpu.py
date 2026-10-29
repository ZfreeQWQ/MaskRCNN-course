"""Checks on actual course train samples, no GPU and no test labels."""
from pathlib import Path

import mmcv
import numpy as np
import torch

from rgbd.course_data import LoadCourseRGBDImage, CourseRGBDCocoDataset
from rgbd.data import PackRGBDDetInputs
from run_course_experiment import parser, build_config, component_state


def main():
    root=Path('/workspace/data/course_blender_20261013/samples')
    rgb=next((root/'rgb/train').glob('*.png'))
    depth=root/'depth/train'/rgb.name
    raw=mmcv.imread(str(depth),flag='unchanged')
    loader=LoadCourseRGBDImage()
    def inp(scale,code=0):
        return dict(img_path=str(rgb),depth_file=str(depth),depth_scale_m=scale,depth_invalid_code=code)
    a=loader.transform(inp(.0001))
    b=loader.transform(inp(.0002))
    np.testing.assert_allclose(b['rgbd_depth'],a['rgbd_depth']*2)
    c=loader.transform(inp(.0001,int(raw[0,0])))
    assert not c['rgbd_raw_valid'][0,0] and c['rgbd_depth'][0,0]==0
    assert c['rgbd_label_valid'].all()
    packed=PackRGBDDetInputs().transform(a)
    assert tuple(packed['inputs'].shape)==(5,1024,1024)
    ds=CourseRGBDCocoDataset(data_root=str(root),ann_file='annotations/smoke_train.json',
        data_prefix=dict(img=''),metainfo=dict(classes=('component',)),pipeline=[],filter_cfg=None)
    assert len(ds)==1 and ds.get_data_info(0)['depth_scale_m']==.0001
    args=parser().parse_args(['train','--output','/tmp/course-config-check','--functional-smoke',
        '--data-root',str(root),'--iters','20','--val-interval','20'])
    cfg=build_config(args,[dict(id=1,name='component',supercategory='component')])
    assert cfg.val_evaluator.type=='CocoMetric' and cfg.val_evaluator.proposal_nums==(1,10,100)
    assert not cfg.model.panoptic_head.ignore_void and cfg.model.panoptic_head.num_things_classes==1
    state=component_state({'panoptic_head.cls_embed.weight':torch.arange(81*3).reshape(81,3).float(),
                           'panoptic_head.cls_embed.bias':torch.arange(81).float()})
    assert state['panoptic_head.cls_embed.weight'].shape==(2,3)
    assert state['panoptic_head.cls_embed.bias'][1]==80
    print('Course CPU checks passed: scaling, invalid code, supervision, packing, dataset, category, evaluator, classifier',flush=True)


if __name__=='__main__':
    main()
