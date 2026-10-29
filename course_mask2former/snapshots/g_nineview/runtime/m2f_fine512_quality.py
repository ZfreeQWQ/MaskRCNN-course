"""Same-instance quality gain, trained on train only; no GT at application."""
import copy
import numpy as np
from pycocotools import mask as mu
import m2f_quality_selector as q
from m2f_fine512_geometry import refine


def paired(current, new_raw):
    # Always pair the full current global branch, before tiny insertion.
    _, activity = refine(current, new_raw)
    return [p for p in activity['pairs'] if p['changed']]


def supervised(current, new_raw, annotations, coco):
    result = []
    for p in paired(current, new_raw):
        old, new = current[p['index']], new_raw[p['new_index']]
        anns = sorted((a for a in annotations if a['category_id'] == old['category_id']
                       and not a.get('iscrowd', 0) and not a.get('ignore', 0)), key=lambda a:a['id'])
        gt, old_iou, new_iou = None, 0., 0.
        if anns:
            values = mu.iou([old['segmentation'], new['segmentation']],
                            [coco.annToRLE(a) for a in anns], [0]*len(anns))
            j = int(np.argmax(values[0]))
            if values[0,j] > 0:
                gt, old_iou, new_iou = anns[j], float(values[0,j]), float(values[1,j])
        small = gt is not None and 0 < gt['area'] <= 1024
        result.append(dict(**p, image_id=old['image_id'], features=q.features(old,new),
            gt_id=gt['id'] if gt else None, gt_area=gt['area'] if gt else None,
            old_iou=old_iou, new_iou=new_iou, actual_delta=new_iou-old_iou,
            target_delta=new_iou-old_iou if small else 0.,
            target_kind='same_small_gt_delta' if small else 'identity_zero'))
    return result


def apply(current, new_raw, before, kept, selector):
    assert len(before) == len(kept) <= 100
    pairs = paired(current,new_raw)
    estimates = q.predict([q.features(current[p['index']],new_raw[p['new_index']]) for p in pairs],selector) if pairs else []
    by_index = {p['index']:dict(**p,predicted_delta=float(v),accepted=bool(v>0)) for p,v in zip(pairs,estimates)}
    result, decisions = copy.deepcopy(before), []
    seen = set()
    for j,(old,origin) in enumerate(zip(before,kept)):
        assert origin['origin'] in ('original','new')
        if origin['origin']=='new':
            continue  # Existing tiny additions stay byte-identical.
        index = origin['index']; assert index not in seen; seen.add(index)
        assert old == current[index], 'Final original output identity mismatch'
        if index not in by_index:
            continue
        d = dict(by_index[index],output_index=j); decisions.append(d)
        if d['accepted']:
            result[j]['segmentation'] = copy.deepcopy(new_raw[d['new_index']]['segmentation'])
            if 'bbox' in result[j]:result[j]['bbox'] = [float(v) for v in mu.toBbox(result[j]['segmentation'])]
    for j,(old,new) in enumerate(zip(before,result)):
        assert {k:v for k,v in old.items() if k not in ('segmentation','bbox')} == {k:v for k,v in new.items() if k not in ('segmentation','bbox')}
        if kept[j]['origin']=='new':assert old==new
    return result, dict(decisions=decisions,current_pairs=len(pairs),final_pairs=len(decisions),
        accepted=sum(d['accepted'] for d in decisions),rejected=sum(not d['accepted'] for d in decisions),
        protected_tiny_outputs=sum(p['origin']=='new' for p in kept),new_instances=0,
        score_class_count_order_preserved=True)


def decision_audit(pairs, decisions):
    truth={p['index']:p for p in pairs}; out=[]
    for d in decisions:
        p=truth[d['index']]; assert p['new_index']==d['new_index']
        out.append(dict(**p,**{k:v for k,v in d.items() if k not in p},
                        threshold_crossings=int(sum(p['new_iou']>=x for x in np.arange(.5,1.,.05))-
                                                sum(p['old_iou']>=x for x in np.arange(.5,1.,.05)))))
    return out
