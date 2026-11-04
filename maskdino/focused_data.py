"""Keep the existing full-image transforms; alternate with training-GT-centred native640 crops."""
import random
import numpy as np
from PIL import Image
from scale_data import GeometryData as Base,torch,F,d,geometry,normalize

class GeometryData(Base):
    focus_indices=set()
    def __getitem__(self,i):
        if i not in self.focus_indices:return super().__getitem__(i)
        assert self.split=='train'
        item=self.base.data.item(i,targets=True,flip=False);row=self.index.records[i]
        image=item['image'];masks=item['instances'].gt_masks
        areas=masks.flatten(1).sum(1)
        candidates=torch.where((areas>0)&(areas<=256))[0].tolist()
        if not candidates:candidates=torch.where((areas>0)&(areas<=1024))[0].tolist()
        if not candidates:return super().__getitem__(i)
        rng=random.Random(121+self.epoch*len(self)+i)
        chosen=rng.choice(candidates);pixels=torch.nonzero(masks[chosen]);cy,cx=map(int,pixels[rng.randrange(len(pixels))])
        x=max(0,min(384,cx-320+rng.randint(-96,96)));y=max(0,min(384,cy-320+rng.randint(-96,96)))
        with Image.open(self.index.path(row['depth_file_name'])) as im:raw=np.array(im)
        values,valid,info=geometry(raw.astype(np.float32)*row['depth_scale_m'],raw!=row['depth_invalid_code'],self.cameras[row['id']],row['id'])
        geom=torch.from_numpy(normalize(values,valid,self.stats))
        # Compute camera geometry at native resolution, then transform all channels and labels together.
        image=image[:,y:y+640,x:x+640];geom=geom[:,y:y+640,x:x+640];masks=masks[:,y:y+640,x:x+640]
        assert masks[chosen].any() and image.shape[-2:]==(640,640)
        flip=rng.random()<.5
        if flip:image=image.flip(-1);geom=geom.flip(-1);masks=masks.flip(-1)
        image=F.interpolate(image[None],(2048,2048),mode='bilinear',align_corners=False)[0]
        geom=F.interpolate(geom[None],(2048,2048),mode='nearest')[0]
        masks=F.interpolate(masks[None].float(),(2048,2048),mode='nearest')[0].bool();masks=masks[masks.flatten(1).any(1)]
        ins=d.Instances((2048,2048));ins.gt_masks=masks;ins.gt_classes=torch.zeros(len(masks),dtype=torch.long);ins.gt_boxes=d.BitMasks(masks).get_bounding_boxes()
        item.update(image=torch.cat((image,geom)),instances=ins,height=2048,width=2048,_focused_crop=True,_geometry_info=info,_focused_crop_meta={'x':x,'y':y,'flip':flip,'chosen_instance':chosen,'chosen_native_area':int(areas[chosen])})
        return item
