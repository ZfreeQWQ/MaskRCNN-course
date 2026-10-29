"""Prediction-only, reciprocal 512-view geometry refinement; no GT API."""
import copy
import math
import numpy as np
from pycocotools import mask as mu


def refine(before, new_raw):
    assert len(before) <= 100
    result = copy.deepcopy(before)
    old_indices = [i for i, r in enumerate(before) if 0 < int(mu.area(r['segmentation'])) <= 1024]
    new_indices = [i for i, r in enumerate(new_raw) if 0 < int(mu.area(r['segmentation'])) <= 1024]
    for r in before + new_raw:
        assert r['segmentation']['size'] == [1024, 1024]
        assert math.isfinite(r['score']) and 0 <= r['score'] <= 1
    pairs = []
    for category in sorted({before[i]['category_id'] for i in old_indices}):
        oi = [i for i in old_indices if before[i]['category_id'] == category]
        ni = [i for i in new_indices if new_raw[i]['category_id'] == category]
        if not ni:
            continue
        ious = mu.iou([before[i]['segmentation'] for i in oi],
                      [new_raw[i]['segmentation'] for i in ni], [0] * len(ni))
        # Deterministic nearest neighbours; tie by raw new score then index,
        # and by original output index on the reverse mapping.
        best_new = [min(range(len(ni)), key=lambda j: (-float(ious[k, j]), -new_raw[ni[j]]['score'], ni[j]))
                    for k in range(len(oi))]
        best_old = [min(range(len(oi)), key=lambda k: (-float(ious[k, j]), oi[k]))
                    for j in range(len(ni))]
        for k, j in enumerate(best_new):
            if best_old[j] != k or float(ious[k, j]) < .5:
                continue
            index, source = oi[k], ni[j]
            changed = before[index]['segmentation'] != new_raw[source]['segmentation']
            if changed:
                result[index]['segmentation'] = copy.deepcopy(new_raw[source]['segmentation'])
                if 'bbox' in result[index]:
                    result[index]['bbox'] = [float(v) for v in mu.toBbox(result[index]['segmentation'])]
            pairs.append(dict(index=index, new_index=source, pair_iou=float(ious[k, j]), changed=changed))
    assert len(result) == len(before)
    assert all({k:v for k,v in a.items() if k not in ('segmentation', 'bbox')} ==
               {k:v for k,v in b.items() if k not in ('segmentation', 'bbox')} for a,b in zip(result, before))
    assert len({p['index'] for p in pairs}) == len({p['new_index'] for p in pairs}) == len(pairs)
    return result, dict(pairs=sorted(pairs, key=lambda p:p['index']), changed=sum(p['changed'] for p in pairs),
                        old_eligible=len(old_indices), new_eligible=len(new_indices),
                        scores_count_order_preserved=True, new_instances=0)
