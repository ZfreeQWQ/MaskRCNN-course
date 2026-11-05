"""Portable confidence head for frozen, unnormalized43-feature inference caches.

This only scores already predicted masks. The feature producer is the frozen
MaskDINO/native refinement pipeline; labels are not inputs to this interface.
"""
import copy,json
from pathlib import Path
import numpy as np
import torch

class FrozenFeatureScorer:
    def __init__(self,weights_path=None,device='cpu'):
        path=Path(weights_path) if weights_path else Path(__file__).parent/'artifacts'/'soft_quality_weights.pth'
        ck=torch.load(path,map_location='cpu',weights_only=False)
        self.features=tuple(ck['features']);assert len(self.features)==43 and len(ck['thresholds'])==10
        self.mean=np.asarray(ck['mean'],np.float32);self.std=np.asarray(ck['std'],np.float32);assert (self.std>0).all()
        self.device=device
        self.model=torch.nn.Sequential(torch.nn.Linear(43,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,10),torch.nn.Sigmoid()).to(device)
        self.model.load_state_dict(ck['model'],strict=True);self.model.eval()

    @torch.inference_mode()
    def score(self,payload,provenance):
        assert tuple(provenance['features'])==self.features
        result=copy.deepcopy(payload['raw_predictions']);positions=payload['positions']
        x=np.asarray(payload['features'],np.float32).reshape(-1,len(self.features))
        assert x.shape[0]==len(positions) and len(set(positions))==len(positions) and np.isfinite(x).all()
        assert len(result)<=100 and all(0<=i<len(result) for i in positions)
        if len(x):
            assert np.allclose(x[:,0],[result[i]['score'] for i in positions],rtol=0,atol=1e-7),'Expected original, unscored confidence'
            q=self.model(torch.from_numpy((x-self.mean)/self.std).to(self.device)).mean(1).cpu().numpy()
            for i,quality in zip(positions,q):result[i]['score']*=float(quality)
        for p in result:
            assert p['category_id']==1 and p['segmentation']['size']==[1024,1024]
            assert isinstance(p['segmentation']['counts'],str) and 0<=p['score']<=1
        return sorted(result,key=lambda p:p['score'],reverse=True)

    def score_file(self,feature_path,provenance_path=None):
        path=Path(feature_path);contract=Path(provenance_path) if provenance_path else path.parent.parent/'provenance.json'
        return self.score(json.loads(path.read_text()),json.loads(contract.read_text()))
