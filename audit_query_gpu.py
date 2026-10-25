"""Exact zero-residual GPU parity, fixed real val input, no metrics tuning."""
import json
from pathlib import Path
import torch
from detectron2.modeling import build_model
from course.common import atomic_json, configuration
from course.experiments import mapper


def main():
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parent
    weights=root/'runs/optimization_v2/d01_geometry_multiscale/model_final_inference.pth'
    loaded=torch.load(weights,map_location="cpu",weights_only=False)["model"]
    configs=["i00_control","j01_p2_queries","j02_geometry_queries"]
    models=[]
    for name in configs:
        cfg=configuration(f"configs/screen/{name}.yaml")
        model=build_model(cfg).eval()
        missing,unexpected=model.load_state_dict(loaded,strict=False)
        assert not unexpected and all(k.startswith("sem_seg_head.predictor.query_initializer.") for k in missing)
        models.append(model)
    records=json.loads((root/'data_cache/index.json').read_text())["val"]
    records=[records[0],records[len(records)//2]]
    transform=mapper(configuration("configs/screen/i00_control.yaml"),False)
    rows=[]
    with torch.inference_mode():
        for record in records:
            image=transform(record)
            reference=models[0]([image])[0]["instances"]
            for name,model in zip(configs[1:],models[1:]):
                actual=model([image])[0]["instances"]
                for field in ("pred_masks","scores","pred_classes","pred_boxes"):
                    a,b=getattr(reference,field),getattr(actual,field)
                    if hasattr(a,"tensor"): a,b=a.tensor,b.tensor
                    torch.testing.assert_close(a,b,rtol=0,atol=0)
                rows.append(dict(config=name,image_id=record["image_id"],exact_masks=True,exact_scores=True))
    output=root/'runs/screen_v2_gpu_audit.json'
    atomic_json(output,dict(status="passed",comparisons=rows,
                original_parameters=sum(p.numel() for p in models[0].parameters()),
                candidate_parameters={n:sum(p.numel() for p in m.parameters()) for n,m in zip(configs[1:],models[1:])}))
    print(output.read_text(),flush=True)


if __name__=="__main__": main()
