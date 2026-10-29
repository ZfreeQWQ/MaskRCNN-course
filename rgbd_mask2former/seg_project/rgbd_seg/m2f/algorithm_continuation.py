"""Optimizer-preserving continuation with an explicitly extended LR schedule."""
import json
from pathlib import Path
import torch
from detectron2.engine import HookBase
from detectron2.utils import comm
from .algorithm_trainer import AlgorithmTrainer


def audit_optimizer(expected, actual):
    if len(expected["param_groups"]) != len(actual["param_groups"]) or len(expected["state"]) != len(actual["state"]):
        raise AssertionError("Optimizer group/state count changed on resume")
    old_ids = [p for g in expected["param_groups"] for p in g["params"]]
    new_ids = [p for g in actual["param_groups"] for p in g["params"]]
    if len(old_ids) != len(new_ids):
        raise AssertionError("Optimizer parameter count changed")
    populated = [(a,b) for a,b in zip(old_ids,new_ids) if a in expected["state"]]
    if not populated:
        raise AssertionError("No optimizer moments to preserve")
    checked = []
    for index in sorted({0, len(populated)//2, len(populated)-1}):
        old, new = populated[index]
        first, second = expected["state"][old], actual["state"][new]
        if set(first) != set(second):
            raise AssertionError("Optimizer state fields changed")
        for key, value in first.items():
            current = second[key]
            if isinstance(value, torch.Tensor):
                if not torch.equal(value.cpu(), current.detach().cpu()):
                    raise AssertionError("Restored optimizer tensor differs: " + key)
            elif value != current:
                raise AssertionError("Restored optimizer scalar differs: " + key)
        checked.append({"parameter_position": index, "state_fields": sorted(first),
                        "step": float(first["step"])})
    return {"valid": True, "state_count": len(expected["state"]),
            "parameter_groups": len(expected["param_groups"]), "checked_moments": checked}


class ContinuationWarmup(HookBase):
    def before_step(self):
        seed = self.trainer.continuation_seed
        elapsed = self.trainer.iter - seed["completed_iterations"]
        # New long schedule deliberately restores a useful LR. Ramp over 100 updates;
        # do not reset moments, global iteration, or base LR ratios between modules.
        if 0 <= elapsed < 100:
            scale = .1 + .9 * (elapsed + 1) / 100
            for group, target in zip(self.trainer._trainer.optimizer.param_groups, self.trainer.scheduler.get_lr()):
                group["lr"] = target * scale


class ContinuationTrainer(AlgorithmTrainer):
    def build_hooks(self):
        result = super().build_hooks()
        result.insert(0, ContinuationWarmup())
        return result

    def resume_or_load(self, resume=True):
        if not resume:
            raise ValueError("Continuation requires --resume and a staged full checkpoint")
        output = Path(self.cfg.OUTPUT_DIR)
        self.continuation_seed = json.loads((output/"resume_seed.json").read_text())
        source = Path(self.checkpointer.get_checkpoint_file())
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        expected = checkpoint["trainer"]["_trainer"]["optimizer"]
        super().resume_or_load(resume=True)
        if self.start_iter != checkpoint["iteration"]+1:
            raise AssertionError("Global resume iteration not restored")
        report = audit_optimizer(expected, self._trainer.optimizer.state_dict())
        report.update(rank=comm.get_rank(), checkpoint=str(source), start_iter=self.start_iter,
                      target_iter=self.max_iter, scheduler_last_epoch=self.scheduler.last_epoch,
                      lr_schedule="extended horizon; 100-update transition warmup; moments preserved")
        if self.scheduler.last_epoch != self.start_iter:
            raise AssertionError("LR scheduler iteration was not restored")
        records = comm.gather(report, dst=0)
        if comm.is_main_process():
            (output/"optimizer_resume_audit.json").write_text(json.dumps({"valid": True, "ranks": records}, indent=2))
