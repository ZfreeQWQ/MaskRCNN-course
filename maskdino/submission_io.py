"""Strict COCO compressed-RLE CSV export, including complete mask round trips."""
import csv
import json
import math
from pathlib import Path


def export(predictions, ids, path):
    import numpy as np
    from pycocotools import mask as mu
    ids = list(ids)
    keys = list(map(str, ids))
    assert len(keys) == len(set(keys)), 'Duplicate image IDs'
    grouped = {key: [] for key in keys}
    for pred in predictions:
        key = str(pred['image_id'])
        assert key in grouped, 'Prediction not present in template'
        item = {name: pred[name] for name in ('category_id', 'score', 'segmentation')}
        assert type(item['category_id']) is int and item['category_id'] == 1
        assert math.isfinite(item['score']) and 0 <= item['score'] <= 1
        seg = item['segmentation']
        assert set(seg) == {'size', 'counts'} and seg['size'] == [1024, 1024]
        assert isinstance(seg['counts'], str) and seg['counts']
        seg['counts'].encode('ascii')
        decoded = mu.decode(seg)
        encoded = mu.encode(np.asfortranarray(decoded))
        assert list(decoded.shape) == [1024, 1024] and encoded['counts'].decode('ascii') == seg['counts']
        grouped[key].append(item)
    assert all(len(p) <= 100 for p in grouped.values())
    path = Path(path)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'PredictionString'])
        for image_id, key in zip(ids, keys):
            writer.writerow([image_id, json.dumps(grouped[key], separators=(',', ':'))])
    csv.field_size_limit(20*1024*1024)
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ['image_id', 'PredictionString']
        rows = list(reader)
    assert [row['image_id'] for row in rows] == keys
    for row in rows:
        assert json.loads(row['PredictionString']) == grouped[row['image_id']]
        if not grouped[row['image_id']]:
            assert row['PredictionString'] == '[]'
    return dict(rows=len(rows), instances=sum(map(len, grouped.values())),
                empty_images=sum(not p for p in grouped.values()),
                max_instances_per_image=max(map(len, grouped.values()), default=0),
                all_rles_round_trip=True, csv_readback_exact=True)
