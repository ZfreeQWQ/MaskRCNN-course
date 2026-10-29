"""CPU integration test: threshold variants preserve RLEs and empty-image coverage."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

from common import write_json


class ThresholdExportTest(unittest.TestCase):
    def test_complete_variants_and_empty_images(self):
        project=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='threshold_export_test_',dir=project) as temporary:
            folder=Path(temporary)
            images=[dict(id=i,height=1024,width=1024,split='test',file_name=f'{i}.png') for i in range(1,2113)]
            write_json(folder/'metadata.json',dict(images=images))
            mask=np.zeros((1024,1024),dtype=np.uint8)
            mask[20:24,30:35]=1
            rle=mask_util.encode(np.asfortranarray(mask))
            rle['counts']=rle['counts'].decode('ascii')
            predictions=[dict(image_id=i,category_id=1,score=s,bbox=[30.,20.,5.,4.],segmentation=rle)
                         for i,s in [(1,.8),(2,.2)]]
            write_json(folder/'master.json',predictions)
            write_json(folder/'manifest.json',dict(split='test',smoke=False,score_threshold=0,
                                                  image_ids=[im['id'] for im in images],num_instances=2))
            write_json(folder/'val.json',dict(split='val',baseline_matches_checkpoint=True,
                                             selected_threshold=0,results=[dict(threshold=t) for t in [0,.05,.1,.3,.5]]))
            with (folder/'template.csv').open('w',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['image_id','PredictionString'])
                writer.writerows((im['id'],'[]') for im in reversed(images))
            subprocess.run([sys.executable,str(Path(__file__).with_name('export_threshold_submissions.py')),
                            '--metadata',str(folder/'metadata.json'),'--predictions',str(folder/'master.json'),
                            '--manifest',str(folder/'manifest.json'),'--template',str(folder/'template.csv'),
                            '--val-report',str(folder/'val.json'),'--output',str(folder/'output'),'--workers','2'],
                           check=True,capture_output=True,text=True)
            report=json.loads((folder/'output/submissions.json').read_text())
            self.assertEqual(len(report['submissions']),5)
            for variant in report['submissions']:
                with open(variant['csv'],newline='') as stream:
                    rows=list(csv.DictReader(stream))
                self.assertEqual(len(rows),2112)
                self.assertEqual([int(row['image_id']) for row in rows],list(range(2112,0,-1)))
                cells={int(row['image_id']):json.loads(row['PredictionString']) for row in rows}
                self.assertEqual(cells[2112],[])
                self.assertEqual(len(cells[1]),1)
                self.assertEqual(len(cells[2]),int(variant['threshold']<=.2))
                np.testing.assert_array_equal(mask_util.decode(cells[1][0]['segmentation']),mask)
                self.assertEqual(variant['num_instances'],2 if variant['threshold']<=.2 else 1)
                self.assertTrue(variant['submission_ready'])
                self.assertEqual(variant['null_counts'],{'image_id':0,'PredictionString':0})


if __name__=='__main__':
    unittest.main()
