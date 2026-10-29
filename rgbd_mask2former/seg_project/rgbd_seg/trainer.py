import json
import math
import shutil
import weakref
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, HookBase, hooks
from detectron2.evaluation import COCOEvaluator
from detectron2.utils import comm
from detectron2.utils.events import CommonMetricPrinter, JSONWriter
from .checkpoint import RGBDCheckpointer
from .data import RGBDMapper
from .metrics import json_safe


def evaluation_dataset_name(cfg, dataset_name):
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
    """Select by validation segm/APs and break ties with segm/AP; retain full results/predictions."""
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
        # Keep periodic model/optimizer/scheduler checkpoint as the normal resume target.
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
    def state_dict(self):
        state = super().state_dict()
        # TrainerBase increments iter after the final step before after_train hooks.
        state["iteration"] = min(self.iter, self.max_iter - 1)
        return state

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=RGBDMapper(cfg, is_train=True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, evaluation_dataset_name(cfg, dataset_name), mapper=RGBDMapper(cfg, is_train=False))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        dataset_name = evaluation_dataset_name(cfg, dataset_name)
        return COCOEvaluator(dataset_name, tasks=("bbox", "segm"), distributed=True,
                             output_dir=output_folder or str(Path(cfg.OUTPUT_DIR) / ("inference_subset" if cfg.PROJECT.EVAL_LIMIT else "inference")),
                             use_fast_impl=False, max_dets_per_image=100)

    def build_hooks(self):
        self.checkpointer = RGBDCheckpointer(self.model, self.cfg.OUTPUT_DIR,
                                            trainer=weakref.proxy(self), depth_init=self.cfg.PROJECT.DEPTH_INIT)
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
        # JSON is sufficient and avoids imposing optional tensorboard dependencies.
        return [CommonMetricPrinter(self.max_iter), JSONWriter(str(Path(self.cfg.OUTPUT_DIR) / "metrics.json"))]
