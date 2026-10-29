#!/usr/bin/env python3
"""GPU regression check for rgbd_seg.m2f.checkpoint.M2FCheckpointer's query-embedding padding.

Guards against the crash found when NUM_OBJECT_QUERIES != 100 (the checkpoint's pretrained
query count): loading used to hard-crash with "size mismatch for
sem_seg_head.predictor.query_feat.weight" because the checkpoint's `static_query` key is
renamed to `query_feat` deep inside nn.Module.load_state_dict's recursion, AFTER fvcore's
shape-mismatch pre-check already let it through unmodified. See checkpoint.py's module
docstring for the full story.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rgbd_seg.m2f.config import get_cfg
from rgbd_seg.m2f.checkpoint import M2FCheckpointer
from detectron2.modeling import build_model


def build(num_queries):
    cfg = get_cfg()
    cfg.merge_from_file('seg_project/third_party/Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml')
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = num_queries
    cfg.freeze()
    return build_model(cfg)


def main():
    weights = 'seg_project/weights/maskformer2_R50_bs16_50ep.pkl'

    # 1. 200 queries (what the project's configs actually use) must load without crashing.
    model = build(200)
    M2FCheckpointer(model).load(weights)  # would previously raise RuntimeError here
    qf = model.sem_seg_head.predictor.query_feat.weight.detach()
    qe = model.sem_seg_head.predictor.query_embed.weight.detach()
    assert qf.shape == (200, 256) and qe.shape == (200, 256)

    # 2. The first 100 rows must be the actual pretrained values, not random.
    import pickle
    raw = pickle.load(open(weights, 'rb'), encoding='latin1')['model']
    ref_feat = torch.as_tensor(raw['sem_seg_head.predictor.static_query.weight'])
    ref_embed = torch.as_tensor(raw['sem_seg_head.predictor.query_embed.weight'])
    torch.testing.assert_close(qf[:100].cpu(), ref_feat)
    torch.testing.assert_close(qe[:100].cpu(), ref_embed)
    print("first 100 query rows match pretrained checkpoint exactly: OK")

    # 3. The extra 100 rows must NOT all be zero/degenerate (i.e. still carry a real random
    # init, not an accidental zero-fill) and must differ row-to-row.
    extra_feat = qf[100:]
    assert extra_feat.abs().sum() > 0
    assert not torch.allclose(extra_feat[0], extra_feat[1])
    print("extra 100 query rows retain their own (non-degenerate) random init: OK")

    # 4. Exactly 100 queries (matches checkpoint) must still load cleanly (no-op path).
    model100 = build(100)
    M2FCheckpointer(model100).load(weights)
    torch.testing.assert_close(model100.sem_seg_head.predictor.query_feat.weight.detach().cpu(), ref_feat)
    print("exact 100-query match (no padding needed) still loads correctly: OK")

    # 5. Fewer than 100 queries must truncate, not crash.
    model50 = build(50)
    M2FCheckpointer(model50).load(weights)
    torch.testing.assert_close(model50.sem_seg_head.predictor.query_feat.weight.detach().cpu(), ref_feat[:50])
    print("truncation to 50 queries works: OK")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
