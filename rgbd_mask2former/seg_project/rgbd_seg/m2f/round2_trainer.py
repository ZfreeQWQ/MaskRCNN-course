"""D+H evaluation uses the actual refinement branch for checkpoint selection."""
import json
from pathlib import Path

from detectron2.evaluation import COCOEvaluator, DatasetEvaluator, inference_on_dataset
from detectron2.utils import comm

from .trainer import Trainer, evaluation_dataset_name
from .round2_inference import PairedInference
from ..metrics import json_safe


class PairedEvaluator(DatasetEvaluator):
    def __init__(self, cfg, dataset_name):
        self.folder = Path(cfg.OUTPUT_DIR) / ("inference_subset" if cfg.PROJECT.EVAL_LIMIT else "inference")
        self.paired = cfg.PROJECT.HIRES_MASK.ENABLED
        options = dict(tasks=("segm",), distributed=True, use_fast_impl=False, max_dets_per_image=100)
        self.refined = COCOEvaluator(dataset_name, output_dir=str(self.folder), **options)
        self.coarse = COCOEvaluator(dataset_name, output_dir=str(self.folder) + "_coarse", **options) if self.paired else None

    def reset(self):
        self.records = []
        self.refined.reset()
        if self.coarse is not None:
            self.coarse.reset()

    def process(self, inputs, outputs):
        self.refined.process(inputs, [{"instances": o["instances"]} for o in outputs])
        if self.coarse is not None:
            self.coarse.process(inputs, [{"instances": o["coarse_instances"]} for o in outputs])
        self.records.extend(o["diagnostics"] for o in outputs)

    def evaluate(self):
        results = self.refined.evaluate()
        coarse_results = self.coarse.evaluate() if self.coarse is not None else None
        records = comm.gather(self.records, dst=0)
        if not comm.is_main_process():
            return {}
        records = [record for shard in records for record in shard]
        ids = [r["image_id"] for r in records]
        if len(ids) != len(set(ids)):
            raise AssertionError("Validation image IDs repeated across ranks")
        self.folder.mkdir(parents=True, exist_ok=True)
        diagnostics = {"images": len(records), "ranking": "mask-first",
                       "selection_branch": "refined" if self.paired else "coarse",
                       "gate": "predicted original-coordinate area in (0,1024]",
                       "refined_queries": sum(r["refined_queries"] for r in records),
                       "changed_pixels": sum(r["changed_pixels"] for r in records),
                       "records": sorted(records, key=lambda r: r["image_id"])}
        (self.folder / "round2_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
        if coarse_results:
            results["coarse_segm"] = coarse_results["segm"]
        (self.folder / "paired_metrics.json").write_text(json.dumps(json_safe(results), indent=2, allow_nan=False))
        return results


class Round2Trainer(Trainer):
    head_lr_multiplier = 10.0
    inference_chunk_size = 4

    @classmethod
    def build_optimizer(cls, cfg, model):
        optimizer = super().build_optimizer(cfg, model)
        head = getattr(model.criterion, "hires_head", None)
        head_ids = {id(p) for p in head.parameters()} if head is not None else set()
        for group in optimizer.param_groups:
            if any(id(p) in head_ids for p in group["params"]):
                group["lr"] *= cls.head_lr_multiplier
        return optimizer

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        if evaluators is not None or len(cfg.DATASETS.TEST) != 1:
            raise ValueError("Round2 requires its paired evaluator and one validation dataset")
        dataset_name = cfg.DATASETS.TEST[0]
        loader = cls.build_test_loader(cfg, dataset_name)
        name = evaluation_dataset_name(cfg, dataset_name)
        evaluator = PairedEvaluator(cfg, name)
        return inference_on_dataset(PairedInference(model, cls.inference_chunk_size), loader, evaluator)
