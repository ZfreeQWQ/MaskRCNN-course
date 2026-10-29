"""MP diagnostics plus the already validated mask-first full-val evaluator."""
import json
from pathlib import Path

from detectron2.engine import HookBase
from detectron2.utils import comm
from .round2_trainer import Round2Trainer


class MPDiagnostics(HookBase):
    def after_step(self):
        if (self.trainer.iter + 1) % 20 and self.trainer.iter + 1 != self.trainer.max_iter:
            return
        model = self.trainer.model.module if hasattr(self.trainer.model, "module") else self.trainer.model
        stats = model.sem_seg_head.predictor.last_pilot_stats
        embedding = model.sem_seg_head.predictor.label_enc.weight
        records = comm.gather(dict(stats or {}, rank=comm.get_rank(),
                                   label_embedding_norm=float(embedding.norm()),
                                   label_gradient_norm=float(embedding.grad.norm()) if embedding.grad is not None else None), dst=0)
        if comm.is_main_process():
            target = Path(self.trainer.cfg.OUTPUT_DIR) / "mp_diagnostics.jsonl"
            with target.open("a") as stream:
                stream.write(json.dumps({"completed_iterations": self.trainer.iter + 1, "ranks": records}) + "\n")


class MPTrainer(Round2Trainer):
    @classmethod
    def build_optimizer(cls, cfg, model):
        optimizer = super().build_optimizer(cfg, model)
        new_ids = {id(p) for p in model.sem_seg_head.predictor.label_enc.parameters()}
        for group in optimizer.param_groups:
            if any(id(p) in new_ids for p in group["params"]):
                group["lr"] *= 10
        return optimizer

    def build_hooks(self):
        result = super().build_hooks()
        # Before evaluation, which switches the decoder's last stats to eval mode.
        result.insert(0, MPDiagnostics())
        return result
