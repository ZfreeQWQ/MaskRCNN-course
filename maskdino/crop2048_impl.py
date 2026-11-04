from refine_inference_20261103 import *
@torch.inference_mode()
def inference(model,data,i,kind,base):
    item=data.item(i,False);image_id=item['image_id']
    if kind=='scale2560':
        image=item['image']
        item['image']=torch.cat((F.interpolate(image[None,:3],(2560,2560),mode='bilinear',align_corners=False)[0],F.interpolate(image[None,3:],(2560,2560),mode='nearest')[0]))
        return encode_pred(model([item])[0]['instances'].to('cpu'),image_id)
    assert kind=='crop640'
    native=item['image'];extra=[]
    # Geometry was computed once in the original camera frame. Crop all channels together.
    for y in (0,384):
        for x in (0,384):
            view=native[:,y:y+640,x:x+640]
            image=torch.cat((F.interpolate(view[None,:3],(2048,2048),mode='bilinear',align_corners=False)[0],F.interpolate(view[None,3:],(2048,2048),mode='nearest')[0]))
            pred=model([dict(image=image,height=640,width=640,image_id=image_id)])[0]['instances'].to('cpu')
            for mask,s,cls in zip(pred.pred_masks,pred.scores,pred.pred_classes):
                assert int(cls)==0
                m=mask.numpy();area=int(m.sum())
                if not 0<area<=2048:continue
                yy,xx=np.nonzero(m)
                if (x>0 and xx.min()<8) or (y>0 and yy.min()<8) or (x+640<1024 and xx.max()>=632) or (y+640<1024 and yy.max()>=632):continue
                full=np.zeros((1024,1024),np.uint8);full[y:y+640,x:x+640]=m
                extra.append(dict(image_id=image_id,category_id=1,score=float(s),segmentation=enc(full)))
    return merge(base,extra)
