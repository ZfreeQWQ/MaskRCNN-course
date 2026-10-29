import json
import tempfile
import unittest
from pathlib import Path
from run_round1 import merge


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.metadata=self.root/'metadata.json'
        self.metadata.write_text(json.dumps({'images':[{'id':i,'height':2,'width':2} for i in (1,2)]}))
        self.shards=[]
        for i in range(2):
            path=self.root/f'shard{i}.json'
            # Empty predictions are intentional: manifests must prove coverage.
            path.write_text('[]')
            m=dict(split='val',smoke=False,image_ids=[i+1],num_images=1,num_instances=0,
                   num_shards=2,shard_index=i,metadata=str(self.metadata),weights='fixed',config='fixed',
                   score_threshold=0,tta=False,tiling=False,options={'size':1024},variant='legacy',
                   code_sha256='fixed',wall_seconds=1,peak_allocated_MiB=1)
            Path(str(path)+'.manifest.json').write_text(json.dumps(m))
            self.shards.append(path)

    def tearDown(self): self.temp.cleanup()

    def change(self,**kwargs):
        path=Path(str(self.shards[1])+'.manifest.json')
        data=json.loads(path.read_text()); data.update(kwargs)
        path.write_text(json.dumps(data))

    def run_merge(self):
        return merge(self.shards,str(self.metadata),'val',self.root/'merged.json')

    def test_zero_detection_images_are_covered(self):
        self.assertEqual(self.run_merge()['image_ids'],[1,2])

    def test_different_resolution_is_rejected(self):
        self.change(options={'size':1280})
        with self.assertRaisesRegex(ValueError,'options'): self.run_merge()

    def test_overlapping_shards_are_rejected(self):
        self.change(image_ids=[1])
        with self.assertRaisesRegex(ValueError,'coverage'): self.run_merge()

    def test_smoke_shards_are_rejected(self):
        self.change(smoke=True)
        with self.assertRaisesRegex(ValueError,'coverage'): self.run_merge()


if __name__=='__main__': unittest.main()
