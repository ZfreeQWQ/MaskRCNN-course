"""Exact zero-residual GPU parity, fixed real val input, no metrics tuning."""
import json
import argparse
from pathlib import Path
import torch
from detectron2.modeling import build_model
from course.common import atomic_json, configuration
from course.experiments import mapper


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--candidates",nargs=2,default=["j01_p2_queries","j02_geometry_queries"])
    parser.add_argument("--output",default="runs/screen_v2_gpu_audit.json")
    parser.add_argument("--reference",default="i00_control")
    args=parser.parse_args()
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parent
    weights=root/'runs/optimization_v2/d01_geometry_multiscale/model_final_inference.pth'
    loaded=torch.load(weights,map_location="cpu",weights_only=False)["model"]
    configs=[args.reference]+args.candidates
    models=[]
    for name in configs:
        cfg=configuration(f"configs/screen/{name}.yaml")
        model=build_model(cfg).eval()
        missing,unexpected=model.load_state_dict(loaded,strict=False)
        assert not unexpected and all(k.startswith(("sem_seg_head.predictor.query_initializer.",
                                                   "sem_seg_head.local_geometry.","sem_seg_head.point_refiner.")) for k in missing)
        models.append(model)
    records=json.loads((root/'data_cache/index.json').read_text())["val"]
    records=[records[0],records[len(records)//2]]
    reference_cfg=configuration(f"configs/screen/{args.reference}.yaml")
    assert all(configuration(f"configs/screen/{n}.yaml").COURSE.TEST_SIZE==reference_cfg.COURSE.TEST_SIZE for n in args.candidates)
    transform=mapper(reference_cfg,False)
    rows=[]
    with torch.inference_mode():
        for record in records:
            image=transform(record)
            assert tuple(image['image'].shape[-2:])==(reference_cfg.COURSE.TEST_SIZE,)*2
            reference=models[0]([image])[0]["instances"]
            for name,model in zip(configs[1:],models[1:]):
                actual=model([image])[0]["instances"]
                for field in ("pred_masks","scores","pred_classes","pred_boxes"):
                    a,b=getattr(reference,field),getattr(actual,field)
                    if hasattr(a,"tensor"): a,b=a.tensor,b.tensor
                    torch.testing.assert_close(a,b,rtol=0,atol=0)
                rows.append(dict(config=name,image_id=record["image_id"],exact_masks=True,exact_scores=True))
    output=root/args.output
    atomic_json(output,dict(status="passed",comparisons=rows,reference=args.reference,input_size=reference_cfg.COURSE.TEST_SIZE,
                original_parameters=sum(p.numel() for p in models[0].parameters()),
                candidate_parameters={n:sum(p.numel() for p in m.parameters()) for n,m in zip(configs[1:],models[1:])}))
    print(output.read_text(),flush=True)


if __name__=="__main__": main()
