"""Identical RGB/mask transforms across both arms; geometry computed before them."""
import json,random
from pathlib import Path
import numpy as np
from PIL import Image
import train_maskdino_extra as t
from geometry_signals import camera_records,geometry,normalize
torch,F,d=t.torch,t.F,t.d

class GeometryData(torch.utils.data.Dataset):
    def __init__(self,split,arm,stats_path,limit=0,side=1536):
        self.side=side
        self.base=t.InputData(split,'rgb',limit);self.index=self.base.index;self.annotations=self.base.annotations
        self.split=split;self.arm=arm;self.epoch=0;self.stats=json.loads(Path(stats_path).read_text())
        self.cameras=camera_records(t.settings.DATA_ROOT,split) if arm=='geometry' else {}
    def __len__(self):return len(self.index.records)
    def item(self,i,targets=False):
        assert not targets or self.split=='train'
        item=self.base.data.item(i,targets=targets,flip=False);r=self.index.records[i];image=item['image']
        geom=None
        if self.arm=='geometry':
            with Image.open(self.index.path(r['depth_file_name'])) as im:raw=np.array(im)
            assert raw.shape==tuple(image.shape[-2:])
            values,masks,info=geometry(raw.astype(np.float32)*r['depth_scale_m'],raw!=r['depth_invalid_code'],self.cameras[r['id']],r['id'])
            geom=torch.from_numpy(normalize(values,masks,self.stats));item['_geometry_info']=info
        if targets:
            masks=item['instances'].gt_masks;rng=random.Random(121+self.epoch*len(self)+i)
            flip=rng.random()<.5;size=round(self.side*rng.uniform(.8,1.2))
            x=rng.randrange(max(0,size-self.side)+1);y=rng.randrange(max(0,size-self.side)+1)
            if flip:
                image=image.flip(-1);masks=masks.flip(-1)
                if geom is not None:geom=geom.flip(-1)
            image=F.interpolate(image[None],size=(size,size),mode='bilinear',align_corners=False)[0]
            if geom is not None:geom=F.interpolate(geom[None],size=(size,size),mode='nearest')[0]
            masks=F.interpolate(masks[None].float(),size=(size,size),mode='nearest')[0].bool() if len(masks) else torch.zeros((0,size,size),dtype=torch.bool)
            image=image[:,y:y+self.side,x:x+self.side];masks=masks[:,y:y+self.side,x:x+self.side]
            pad=(0,self.side-image.shape[-1],0,self.side-image.shape[-2])
            image=F.pad(image,pad,value=128.);masks=F.pad(masks,pad,value=False)
            if geom is not None:geom=F.pad(geom[:,y:y+self.side,x:x+self.side],pad,value=0.)
            masks=masks[masks.flatten(1).any(1)]
            ins=d.Instances((self.side,self.side));ins.gt_masks=masks;ins.gt_classes=torch.zeros(len(masks),dtype=torch.long);ins.gt_boxes=d.BitMasks(masks).get_bounding_boxes()
            item.update(instances=ins,height=self.side,width=self.side)
        else:
            image=F.interpolate(image[None],size=(self.side,self.side),mode='bilinear',align_corners=False)[0]
            if geom is not None:geom=F.interpolate(geom[None],size=(self.side,self.side),mode='nearest')[0]
        item['image']=torch.cat((image,geom)) if geom is not None else image
        return item
    def __getitem__(self,i):return self.item(i,True)
