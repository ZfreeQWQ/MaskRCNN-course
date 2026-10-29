"""New-algorithm optimizer groups and per-rank diagnostic evidence."""
import json
from pathlib import Path
import torch
from detectron2.engine import HookBase
from detectron2.utils import comm
from .round2_trainer import Round2Trainer
from .mpformer_trainer import MPDiagnostics


class FrequencyDiagnostics(HookBase):
    def after_step(self):
        if (self.trainer.iter + 1) % 20 and self.trainer.iter + 1 != self.trainer.max_iter:
            return
        model = self.trainer.model.module if hasattr(self.trainer.model, "module") else self.trainer.model
        fusion = getattr(model.sem_seg_head.pixel_decoder, "frequency_fusion", None)
        if fusion is None:
            return
        norms = {name: float(param.grad.norm()) if param.grad is not None else None
                 for name, param in fusion.named_parameters()}
        records = comm.gather({"rank": comm.get_rank(), "alpha": float(fusion.alpha.detach()),
                               "residual_scale": float(fusion.alpha.detach().tanh()),
                               "gradient_norms": norms,
                               "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated()}, dst=0)
        if comm.is_main_process():
            with (Path(self.trainer.cfg.OUTPUT_DIR) / "frequency_diagnostics.jsonl").open("a") as stream:
                stream.write(json.dumps({"completed_iterations": self.trainer.iter + 1, "ranks": records}, allow_nan=False) + "\n")


class AlgorithmTrainer(Round2Trainer):
    @classmethod
    def build_optimizer(cls, cfg, model):
        optimizer = super().build_optimizer(cfg, model)
        fusion = getattr(model.sem_seg_head.pixel_decoder, "frequency_fusion", None)
        embedding = getattr(model.sem_seg_head.predictor, "label_enc", None)
        new = {id(p) for module in (fusion, embedding) if module is not None for p in module.parameters()}
        for group in optimizer.param_groups:
            if any(id(p) in new for p in group["params"]):
                group["lr"] *= 10
        return optimizer

    def build_hooks(self):
        result = super().build_hooks()
        if self.cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME == "MaskPilotedDecoder":
            result.insert(0, MPDiagnostics())
        result.insert(0, FrequencyDiagnostics())
        return result
