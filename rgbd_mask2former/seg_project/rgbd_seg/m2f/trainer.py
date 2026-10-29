"""DefaultTrainer subclass for Mask2Former on this project's RGB-D data (docs/plan.md §6/§7).

Mirrors rgbd_seg/trainer.py's Mask R-CNN Trainer (evaluation_dataset_name, BestMaskCheckpoint
selection logic reused verbatim) with three Mask2Former-specific swaps:
  - RGBDInstanceDatasetMapper instead of RGBDMapper (bitmask GT, LSJ-style resize+crop, optional
    depth-geometry channels -- see mapper.py).
  - build_optimizer/build_lr_scheduler copied from third_party/Mask2Former/train_net.py's
    Trainer (AdamW, backbone LR multiplier, full-model gradient clipping) -- the plain
    DefaultTrainer defaults (SGD) don't match Mask2Former's design and the official repo
    doesn't export this as an importable function.
  - Checkpointer: always rgbd_seg.m2f.checkpoint.M2FCheckpointer, which conditionally remaps
    backbone.*->backbone.rgb_backbone.* (only when I1's gated backbone is actually in use) and
    pads the 100-query pretrained embeddings up to NUM_OBJECT_QUERIES; see checkpoint.py's
    module docstring for both bugs this guards against.
"""
import copy
import itertools
import json
import math
import shutil
import weakref
from pathlib import Path
from typing import Any, Dict, List, Set

import torch

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, HookBase, hooks
from detectron2.evaluation import COCOEvaluator
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils import comm
from detectron2.utils.events import CommonMetricPrinter, JSONWriter

from .checkpoint import M2FCheckpointer
from .mapper import RGBDInstanceDatasetMapper
from ..metrics import json_safe


def evaluation_dataset_name(cfg, dataset_name):
    """Identical logic to rgbd_seg.trainer.evaluation_dataset_name (duplicated, not imported,
    to keep the Mask R-CNN and Mask2Former pipelines independently editable -- see docs/plan.md
    §9's rationale for parallel, not shared, training stacks)."""
    limit = cfg.PROJECT.EVAL_LIMIT
    if limit <= 0:
        return dataset_name
    name = dataset_name + f"_smoke_{limit}"
    if name in DatasetCatalog.list():
        return name
    root = Path(cfg.PROJECT.DATASET_ROOT)
    output = Path(cfg.OUTPUT_DIR) / "inference_subset"
    path = output / f"instances_val_first_{limit}.json"
    with open(MetadataCatalog.get(dataset_name).json_file) as stream:
        data = json.load(stream)
    images = sorted(data["images"], key=lambda im: im["id"])[:limit]
    ids = {im["id"] for im in images}
    if comm.is_main_process():
        output.mkdir(parents=True, exist_ok=True)
        subset = dict(data)
        subset["images"] = images
        subset["annotations"] = [a for a in data["annotations"] if a["image_id"] in ids]
        path.write_text(json.dumps(subset))
        (output / "SMOKE_ONLY.txt").write_text("Subset metrics are NOT full validation metrics and cannot select a final model.\n")
    comm.synchronize()
    DatasetCatalog.register(name, lambda: [r for r in DatasetCatalog.get(dataset_name) if r["image_id"] in ids])
    MetadataCatalog.get(name).set(json_file=str(path), image_root=str(root), evaluator_type="coco",
                                  thing_classes=["component"], thing_dataset_id_to_contiguous_id={1: 0})
    return name


class BestMaskCheckpoint(HookBase):
    """Identical selection logic to rgbd_seg.trainer.BestMaskCheckpoint (segm/APs first,
    segm/AP tiebreak); duplicated rather than imported for the same reason as above."""
    def __init__(self):
        self.best = (-math.inf, -math.inf)
        self.best_iteration = -1
        self.last_checked_iteration = -1

    def before_train(self):
        path = Path(self.trainer.cfg.OUTPUT_DIR) / "best_metrics.json"
        if path.exists():
            previous = json.loads(path.read_text())
            score = (previous["metrics"]["segm"]["APs"], previous["metrics"]["segm"]["AP"])
            if score > self.best:
                self.best = score
                self.best_iteration = previous["iteration"]

    def state_dict(self):
        return {"best": self.best, "best_iteration": self.best_iteration,
                "last_checked_iteration": self.last_checked_iteration}

    def load_state_dict(self, state):
        self.best = tuple(state["best"])
        self.best_iteration = state["best_iteration"]
        self.last_checked_iteration = state["last_checked_iteration"]

    def _check(self):
        trainer = self.trainer
        results = getattr(trainer, "_last_eval_results", None)
        iteration = min(trainer.iter, trainer.max_iter - 1)
        if not results or self.last_checked_iteration == iteration:
            return
        self.last_checked_iteration = iteration
        segm = results.get("segm", {})
        score = (float(segm.get("APs", float("nan"))), float(segm.get("AP", float("nan"))))
        if not all(math.isfinite(s) and s >= 0 for s in score):
            return
        output = Path(trainer.cfg.OUTPUT_DIR)
        (output / "latest_eval_metrics.json").write_text(json.dumps(
            json_safe({"iteration": iteration, "metrics": results}), indent=2, allow_nan=False))
        if score <= self.best:
            return
        self.best, self.best_iteration = score, iteration
        previous = trainer.checkpointer.get_checkpoint_file()
        trainer.checkpointer.save("best_segm_APs", iteration=iteration)
        if previous:
            trainer.checkpointer.tag_last_checkpoint(Path(previous).name)
        summary = {"iteration": iteration, "completed_iterations": iteration + 1,
                   "selection_metric": "segm/APs", "tie_breaker": "segm/AP", "metrics": results,
                   "checkpoint": "best_segm_APs.pth"}
        (output / "best_metrics.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False))
        source = output / "inference" / "coco_instances_results.json"
        if source.exists():
            shutil.copy2(source, output / "best_val_predictions.json")

    def after_step(self):
        period = self.trainer.cfg.TEST.EVAL_PERIOD
        if period > 0 and (self.trainer.iter + 1) % period == 0 and self.trainer.iter + 1 < self.trainer.max_iter:
            self._check()

    def after_train(self):
        if self.trainer.iter >= self.trainer.max_iter and not self.trainer.cfg.PROJECT.SKIP_FINAL_EVAL:
            self._check()


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        # FOUND VIA A REAL 4-GPU HANG (docs/plan.md's "dsh" screening run, 2026-10-23): when
        # PROJECT.HIRES_MASK is enabled, HiResCriterion only runs rgbd_seg.m2f.hires_mask.
        # HiResMaskHead on matched pairs whose GT is "small" (see criterion.py). If a given
        # rank's local batch happens to contain zero small targets for the main decoder layer in
        # some iteration, that rank's backward graph never touches hires_head's parameters that
        # iteration, while other ranks (whose batches do have small targets) do touch them. Without
        # find_unused_parameters=True, DDP's gradient-bucket reducer assumes every rank produces
        # gradients for the exact same parameter set every iteration; a mismatch deadlocks the
        # NCCL allreduce for that bucket on every rank -- which is what actually happened (the
        # visible symptom was a gloo-backend metrics-gather call 30 minutes later timing out,
        # which is downstream of, not the cause of, the real deadlock in the backward's allreduce).
        # detectron2's create_ddp_model is called with a hardcoded kwarg set inside
        # DefaultTrainer.__init__ with no override hook, so we patch it (module-global lookup,
        # safe and standard for this kind of one-off kwarg injection) before calling super().
        if cfg.PROJECT.HIRES_MASK.ENABLED and comm.get_world_size() > 1:
            import detectron2.engine.defaults as d2_defaults
            if not getattr(d2_defaults.create_ddp_model, "_find_unused_parameters_patch", False):
                original_create_ddp_model = d2_defaults.create_ddp_model

                def _patched_create_ddp_model(model, **kwargs):
                    kwargs.setdefault("find_unused_parameters", True)
                    return original_create_ddp_model(model, **kwargs)
                _patched_create_ddp_model._find_unused_parameters_patch = True
                d2_defaults.create_ddp_model = _patched_create_ddp_model
        super().__init__(cfg)

    def state_dict(self):
        state = super().state_dict()
        state["iteration"] = min(self.iter, self.max_iter - 1)
        return state

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=RGBDInstanceDatasetMapper(cfg, is_train=True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, evaluation_dataset_name(cfg, dataset_name),
                                           mapper=RGBDInstanceDatasetMapper(cfg, is_train=False))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        dataset_name = evaluation_dataset_name(cfg, dataset_name)
        return COCOEvaluator(dataset_name, tasks=("bbox", "segm"), distributed=True,
                             output_dir=output_folder or str(Path(cfg.OUTPUT_DIR) / ("inference_subset" if cfg.PROJECT.EVAL_LIMIT else "inference")),
                             use_fast_impl=False, max_dets_per_image=100)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """Copied from third_party/Mask2Former/train_net.py's Trainer.build_optimizer (not
        importable from the official repo as a standalone function): AdamW, backbone params at
        SOLVER.BACKBONE_MULTIPLIER x base LR, zero weight decay on norm layers / embeddings /
        position-bias tables, optional full-model gradient clipping."""
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {"lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
        norm_module_types = (
            torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm, torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm, torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad or value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if "relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name:
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                      and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                      and clip_norm_val > 0.0)

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    def build_hooks(self):
        # M2FCheckpointer is always correct regardless of PROJECT.USE_DEPTH / NUM_OBJECT_QUERIES
        # (its backbone remap is gated on the model's actual structure at runtime, and its
        # query-embedding padding is a no-op when shapes already match); see its module
        # docstring for why a single universal checkpointer class is used here.
        self.checkpointer = M2FCheckpointer(self.model, self.cfg.OUTPUT_DIR, trainer=weakref.proxy(self))
        result = super().build_hooks()
        if self.cfg.PROJECT.SKIP_FINAL_EVAL:
            for hook in result:
                if isinstance(hook, hooks.EvalHook):
                    hook._eval_after_train = False
        if comm.is_main_process() and not self.cfg.PROJECT.EVAL_LIMIT:
            index = next(i for i, hook in enumerate(result) if isinstance(hook, hooks.EvalHook))
            result.insert(index + 1, BestMaskCheckpoint())
        return result

    def build_writers(self):
        return [CommonMetricPrinter(self.max_iter), JSONWriter(str(Path(self.cfg.OUTPUT_DIR) / "metrics.json"))]
