"""Official Mask2Former trainer + course data, finite guards, async val on a second GPU."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_train_loader
from detectron2.engine import hooks, default_setup
from train_net import Trainer as OfficialTrainer

from course.common import atomic_json, configuration
from course.data import RGBDMapper, register


class Trainer(OfficialTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=RGBDMapper(cfg, training=True))

    def build_hooks(self):
        # Evaluation is explicitly handled by a separate GPU process, not
        # DefaultTrainer's blocking/final evaluator.
        return [hook for hook in super().build_hooks() if not isinstance(hook, hooks.EvalHook)]

    def build_writers(self):
        # Keep logging native to PyTorch/Detectron2. TensorBoard's compatibility
        # layer would import the container's unrelated TensorFlow installation.
        from detectron2.utils.events import CommonMetricPrinter, JSONWriter
        return [CommonMetricPrinter(self.max_iter), JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json"))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/blender_rgbd_r50.yaml")
    parser.add_argument("--cache", default="data_cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-iter", type=int, default=30000)
    parser.add_argument("--eval-gpu", default="1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "status.json").exists() and not args.resume:
        raise RuntimeError("Existing run: use --resume or a new output directory")
    torch.set_num_threads(4)
    index = register(args.cache)
    cfg = configuration(args.config, ["OUTPUT_DIR", str(output), "SOLVER.MAX_ITER", str(args.max_iter),
                                     "SOLVER.CHECKPOINT_PERIOD", "5" if args.smoke_test else "1000",
                                     "DATALOADER.NUM_WORKERS", "0" if args.smoke_test else "4"])
    default_setup(cfg, args)
    state = dict(status="initializing", architecture="Mask2Former-ResNet50-RGBD", step=0,
                 max_iter=args.max_iter, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                 train_images=len(index["train"]), val_images=len(index["val"]),
                 validation_results=[], started_unix=time.time(), amp=cfg.SOLVER.AMP.ENABLED,
                 optimizer="AdamW", base_lr=cfg.SOLVER.BASE_LR,
                 train_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), evaluation_gpu=args.eval_gpu)

    def status(**updates):
        state.update(updates)
        state["updated_unix"] = time.time()
        atomic_json(output / "status.json", state)

    status()
    try:
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        assert trainer.model.backbone.stem.conv1.weight.shape[1] == 4
        assert next(trainer.model.parameters()).dtype == torch.float32
        if not args.resume:
            # Check exact COCO transfer rather than trusting by-name load logs.
            loaded = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu", weights_only=False)["model"]
            actual = trainer.model.state_dict()
            wrong = [key for key, value in loaded.items()
                     if key not in actual or not torch.equal(actual[key].detach().cpu(), value)]
            assert not wrong, "COCO transfer mismatches: " + str(wrong[:10])
            atomic_json(output / "initialization_audit.json", dict(
                checked_tensors=len(loaded), mismatches=wrong,
                new_parameters="single-class classifier; class-dependent criterion buffer rebuilt",
                depth_kernel_initially_zero=bool(torch.count_nonzero(actual["backbone.stem.conv1.weight"][:, 3]) == 0)))
        sources = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                   ("train_rgbd.py", "evaluate_rgbd.py", "course/data.py", "configs/blender_rgbd_r50.yaml")}
        atomic_json(output / "protocol.json", dict(
            torch=torch.__version__, cuda=torch.version.cuda, config=cfg.dump(), source_sha256=sources,
            train_val_never_merged=True, test_used=False,
            weights=json.loads(Path("weights/mask2former_r50_coco_rgbd.json").read_text()),
            final_reporting="final fixed-budget checkpoint on full val; intermediate metrics are monitoring only",
            augmentation="synchronized horizontal flip + scale(0.5,1.5) + 1024 crop/pad; no Copy-Paste",
            rgbd="4-channel early fusion; RGB COCO kernels preserved, depth kernels zero-initialized"))

        class Monitor(hooks.HookBase):
            def __init__(self):
                self.process = None
                self.log_handle = None
                self.result_path = None
                self.started = time.monotonic()

            def collect(self, wait=False):
                if self.process is None:
                    return
                if wait:
                    self.process.wait()
                elif self.process.poll() is None:
                    return
                self.log_handle.close()
                if self.process.returncode:
                    raise RuntimeError("Validation failed; see " + str(self.result_path.with_suffix(".log")))
                result = json.loads(self.result_path.read_text())
                assert all(math.isfinite(result[key]) and 0 <= result[key] <= 1
                           for key in ("mask_ap", "mask_aps", "mask_ap50"))
                state["validation_results"].append(result)
                status(active_evaluation=None)
                print("VALIDATION", json.dumps(result), flush=True)
                self.process = None

            def evaluate(self, weights, samples, label):
                self.result_path = output / (label + ".json")
                self.log_handle = self.result_path.with_suffix(".log").open("w")
                command = [sys.executable, "-u", "evaluate_rgbd.py", "--config", args.config,
                           "--cache", args.cache, "--weights", weights,
                           "--samples", str(samples), "--output", str(self.result_path)]
                self.process = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.eval_gpu),
                                                stdout=self.log_handle, stderr=subprocess.STDOUT)
                status(active_evaluation=label, evaluation_pid=self.process.pid)

            def after_step(self):
                step = self.trainer.iter + 1
                self.collect()
                metrics = {key: float(value[0]) for key, value in self.trainer.storage.latest().items()}
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("Nonfinite training metrics")
                if step == 1:
                    gradient = self.trainer.model.backbone.stem.conv1.weight.grad
                    assert gradient is not None and torch.isfinite(gradient).all()
                    assert torch.count_nonzero(gradient[:, 3]) > 0, "Depth is disconnected"
                    atomic_json(output / "gradient_audit.json", dict(depth_gradient_finite=True,
                        depth_gradient_nonzero=True, mean_absolute_depth_gradient=float(gradient[:, 3].abs().mean())))
                if step % 20 == 0 or step == 1:
                    status(status="training", step=step, sample_presentations=step * cfg.SOLVER.IMS_PER_BATCH,
                           metrics=metrics, elapsed_seconds=time.monotonic() - self.started)
                    print("TRAIN_STEP", step, "METRICS", metrics, flush=True)
                # PeriodicCheckpointer has already published this iteration.
                if not args.smoke_test and step < self.trainer.max_iter and (step == 1000 or step % 5000 == 0):
                    if self.process is None:
                        weights = self.trainer.checkpointer.get_checkpoint_file()
                        self.evaluate(weights, 400 if step == 1000 else 0,
                                      "val400_step1000" if step == 1000 else "val_full_step" + str(step))
                    else:
                        print("EVAL_BUSY_SKIP", step, flush=True)

            def after_train(self):
                if self.trainer.iter < self.trainer.max_iter:
                    return  # Never label an interrupted run complete.
                self.collect(wait=True)
                final = str(output / "model_final.pth")
                assert Path(final).is_file()
                status(status="final_validation", step=self.trainer.max_iter)
                self.evaluate(final, 2 if args.smoke_test else 0,
                              "smoke_val2" if args.smoke_test else "val_full_final")
                self.collect(wait=True)
                status(status="smoke_test_passed" if args.smoke_test else "completed",
                       final_metrics=state["validation_results"][-1])

        trainer.register_hooks([Monitor()])
        trainer.train()
    except BaseException as error:
        status(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
