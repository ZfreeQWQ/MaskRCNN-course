"""Select among all pre-existing compatible local masks; keep global scores."""
import copy
import numpy as np
from pycocotools import mask as mu
import m2f_quality_selector as quality
import m2f_cached_geometry_refine as geometry


def candidates(base, merged):
    if not base or not merged:
        return {}
    areas = np.asarray(mu.area([r["segmentation"] for r in base]))
    local = [j for j in geometry.identify_local(base,merged)
             if 0 < float(mu.area(merged[j]["segmentation"])) <= 1024]
    small = [i for i,a in enumerate(areas) if 0 < a <= 1024]
    if not small or not local:
        return {}
    overlaps = mu.iou([base[i]["segmentation"] for i in small],
                      [merged[j]["segmentation"] for j in local],[0]*len(local))
    result = {}
    for k,i in enumerate(small):
        pairs = [dict(index=i,local_index=j,overlap=float(overlaps[k,l]))
                 for l,j in enumerate(local) if overlaps[k,l]>=.5
                 and base[i]["category_id"]==merged[j]["category_id"]]
        pairs.sort(key=lambda p:(-p["overlap"],-merged[p["local_index"]]["score"],p["local_index"]))
        if pairs: result[i]=pairs
    return result


def supervised(base, merged, annotations, coco):
    eligible=candidates(base,merged)
    valid=sorted((a for a in annotations if not a.get("iscrowd",0) and not a.get("ignore",0)),
                 key=lambda a:a["id"])
    masks=[coco.annToRLE(a) for a in valid]
    bg=mu.iou([r["segmentation"] for r in base],masks,[0]*len(masks)) if base and masks else None
    lg=mu.iou([r["segmentation"] for r in merged],masks,[0]*len(masks)) if merged and masks else None
    pairs=[]; oracle=[]
    for i,g in enumerate(base):
        same=[j for j,a in enumerate(valid) if a["category_id"]==g["category_id"]]
        j=max(same,key=lambda j:(bg[i,j],-j)) if same else None
        before=float(bg[i,j]) if j is not None else 0.
        gt=valid[j] if before>0 else None
        available=eligible.get(i,[])
        after=[float(lg[c["local_index"],j]) if gt is not None else 0. for c in available]
        if gt is not None:
            oracle.append(dict(image_id=g["image_id"],index=i,gt_id=gt["id"],gt_area=gt["area"],
                before=before,single=max(before,after[0]) if after else before,
                multiple=max([before]+after),eligible=len(available)))
        for n,c in enumerate(available):
            pairs.append(dict(**c,image_id=g["image_id"],features=quality.features(g,merged[c["local_index"]]),
                gt_id=gt["id"] if gt else None,gt_area=gt["area"] if gt else None,
                global_iou=before,local_iou=after[n],target_delta=after[n]-before,
                original_single=n==0))
    return pairs,oracle


def oracle_summary(records):
    unique={}
    for r in records:
        key=r["gt_id"]
        if key not in unique:
            unique[key]=dict(gt_id=key,image_id=r["image_id"],area=r["gt_area"],single=0.,multiple=0.)
        unique[key]["single"]=max(unique[key]["single"],r["single"])
        unique[key]["multiple"]=max(unique[key]["multiple"],r["multiple"])
    values=[v for v in unique.values() if v["area"]<=1024]
    thresholds=np.linspace(.5,.95,10)
    tiny_extra=[v for v in values if v["area"]<=256 and any(v["single"]<t<=v["multiple"] for t in thresholds)]
    mean=float(np.mean([v["multiple"]-v["single"] for v in values])) if values else 0.
    assert all(v["multiple"]>=v["single"] for v in values)
    groups={}
    for name,subset in (("small",values),("tiny",[v for v in values if v["area"]<=256])):
        groups[name]=dict(count=len(subset),mean_iou_increment=(
            float(np.mean([v["multiple"]-v["single"] for v in subset])) if subset else 0.),
            coverage={str(float(t)):dict(single=int(sum(v["single"]>=float(t) for v in subset)),
                multiple=int(sum(v["multiple"]>=float(t) for v in subset))) for t in thresholds})
    return dict(gate_passed=bool(tiny_extra and mean>0),groups=groups,tiny_extra=tiny_extra,
                unique_small=values,scope="same-global-best-GT, aggregated by unique GT; oracle not AP")


def choose(base,merged,model):
    output=copy.deepcopy(base)
    eligible=candidates(base,merged)
    flat=[p for choices in eligible.values() for p in choices]
    if not flat:return output,[]
    pred=quality.predict([quality.features(base[c["index"]],merged[c["local_index"]]) for c in flat],model)
    scores={(c["index"],c["local_index"]):float(v) for c,v in zip(flat,pred)}
    decisions=[]
    for i,choices in eligible.items():
        chosen=min(choices,key=lambda c:(-scores[i,c["local_index"]],-c["overlap"],
            -merged[c["local_index"]]["score"],c["local_index"]))
        value=scores[i,chosen["local_index"]]
        if value>0:
            output[i]["segmentation"]=copy.deepcopy(merged[chosen["local_index"]]["segmentation"])
            if "bbox" in output[i]:
                output[i]["bbox"]=mu.toBbox(output[i]["segmentation"]).tolist()
        decisions.append(dict(**chosen,predicted_delta=value,accepted=bool(value>0),
            alternatives=len(choices),different_from_max_overlap=chosen["local_index"]!=choices[0]["local_index"]))
    assert len(output)==len(base)
    for a,b in zip(base,output):
        assert {k:v for k,v in a.items() if k not in ("bbox","segmentation")}=={
            k:v for k,v in b.items() if k not in ("bbox","segmentation")}
    return output,decisions


def realized(pairs, model, multiple):
    """Fixed internal diagnostic using same-GT labels, never used to tune."""
    if not pairs:return dict(count=0)
    pred=quality.predict([p["features"] for p in pairs],model)
    groups={}
    for p,v in zip(pairs,pred):
        if multiple or p["original_single"]:
            groups.setdefault((p["image_id"],p["index"]),[]).append((p,float(v)))
    result=[]
    for choices in groups.values():
        # Ordering in pairs already implements overlap/score/index tie breaks.
        p,value=max(choices,key=lambda item:item[1])
        result.append(dict(gt_area=p["gt_area"],accepted=value>0,delta=p["target_delta"],
                           realized_delta=p["target_delta"] if value>0 else 0.))
    out={}
    for name,subset in (("all",result),("tiny",[r for r in result if r["gt_area"] is not None and r["gt_area"]<=256]),
        ("small_257_1024",[r for r in result if r["gt_area"] is not None and 256<r["gt_area"]<=1024]),
        ("large_or_unmatched",[r for r in result if r["gt_area"] is None or r["gt_area"]>1024])):
        out[name]=dict(count=len(subset),harmful_accepted=sum(r["accepted"] and r["delta"]<0 for r in subset),
            beneficial_rejected=sum(not r["accepted"] and r["delta"]>0 for r in subset),
            realized_iou_delta_sum=float(sum(r["realized_delta"] for r in subset)))
    return out
