"""Matched task-checkpoint continuation, fixed small val only, fresh optimizer."""
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
from detectron2.engine import hooks, default_setup
from course.common import atomic_json, configuration, cpu_state_dict
from course.data import register
from course.screen_data import load_panel
from train_rgbd import Trainer


class ScreenTrainer(Trainer):
    @classmethod
    def build_optimizer(cls, cfg, model):
        optimizer = super().build_optimizer(cfg, model)
        initializer = getattr(model.sem_seg_head.predictor, "query_initializer", None)
        if initializer is not None:
            ids = {id(p) for p in initializer.parameters()}
            for group in optimizer.param_groups:
                if any(id(p) in ids for p in group["params"]):
                    assert all(id(p) in ids for p in group["params"])
                    group["lr"] = cfg.SOLVER.BASE_LR * cfg.COURSE.QUERY.LR_MULTIPLIER
        return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--eval-period", type=int, default=1000)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    assert args.max_iter > 0 and args.eval_period > 0
    output = Path(args.output).resolve()
    if (output / "status.json").exists():
        raise RuntimeError("Never overwrite or silently resume a screening run")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    index = register("data_cache")
    records = load_panel(args.panel, index["val"])
    if not args.smoke_test:
        assert len(records) == 200, "Formal screen is exactly the frozen 200-image panel"
        assert args.max_iter == 3000 and args.eval_period == 1000
    overrides = ["OUTPUT_DIR", str(output), "MODEL.WEIGHTS", str(Path(args.weights).resolve()),
                 "SOLVER.MAX_ITER", str(args.max_iter), "SOLVER.CHECKPOINT_PERIOD", str(args.eval_period),
                 "DATALOADER.NUM_WORKERS", "0" if args.smoke_test else "4"]
    cfg = configuration(args.config, overrides)
    assert cfg.COURSE.TEST_SIZE == cfg.INPUT.IMAGE_SIZE == 1024
    assert cfg.SOLVER.IMS_PER_BATCH == 2 and not cfg.SOLVER.AMP.ENABLED
    default_setup(cfg, args)
    state = dict(status="initializing", step=0, additional_steps=args.max_iter,
                 starting_task_training_steps=30000, validation_images=len(records),
                 experiment=cfg.COURSE.NAME, gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                 validation_results=[], started_unix=time.time(), test_used=False, full_val=False)

    def update(**values):
        state.update(values); state["updated_unix"] = time.time()
        atomic_json(output / "status.json", state)

    update()
    try:
        trainer = ScreenTrainer(cfg)
        # Strict weights-only load, NOT resume_or_load: both arms reset optimizer,
        # LR scheduler, iteration and data sampler identically at the task model.
        loaded = torch.load(args.weights, map_location="cpu", weights_only=False)["model"]
        missing, unexpected = trainer.model.load_state_dict(loaded, strict=False)
        expected_new = {k for k in trainer.model.state_dict() if k.startswith("sem_seg_head.predictor.query_initializer.")}
        assert set(missing) == expected_new and not unexpected, (missing, unexpected)
        assert bool(expected_new) == (cfg.COURSE.QUERY.MODE != "off")
        actual = trainer.model.state_dict()
        mismatches = [k for k, v in loaded.items() if not torch.equal(v, actual[k].detach().cpu())]
        assert not mismatches
        assert not trainer.optimizer.state, "Optimizer must be fresh in both arms"
        initial_sha = hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()
        panel_sha = hashlib.sha256(Path(args.panel).read_bytes()).hexdigest()
        atomic_json(output / "initialization_audit.json", dict(weights_sha256=initial_sha,
                    checked_tensors=len(loaded), mismatches=mismatches, optimizer_reset=True,
                    explicitly_new_tensors=sorted(expected_new),
                    source="trained d01 geometry task checkpoint, originally COCO-initialized"))
        del loaded, actual
        revision = json.loads(Path(".source_revision.json").read_text()) if Path(".source_revision.json").exists() else {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
        atomic_json(output / "protocol.json", dict(config=cfg.dump(), source_revision=revision,
            weights_sha256=initial_sha, panel_sha256=panel_sha, seed=cfg.SEED,
            task_steps_before=30000, task_steps_added=args.max_iter,
            optimizer="fresh AdamW; matching official grouping, backbone multiplier and geometry LR",
            comparison="matched control and candidate from identical task weights, budget, seed, sampler, augmentations and losses",
            query_settings=dict(cfg.COURSE.QUERY),
            train_val_never_merged=True, test_used=False, automatic_full_validation=False,
            validation="fixed 200-image scene-stratified screen; not a full-val or convergence claim",
            geometry_assets={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             (Path("data_cache/geometry_v2_intrinsics.json"), Path("data_cache/geometry_v2_stats.json"))}))

        class ScreenMonitor(hooks.HookBase):
            query_inner_gradient_seen = False
            def evaluate(self, weights, step):
                label = f"screen_step{step}"
                result_path = output / (label + ".json")
                update(status="screen_evaluation", active_evaluation=label, step=step)
                # Synchronous same-GPU evaluation: other GPU owns the other arm;
                # no cross-run compute interference, and training RNG/data order
                # cannot be consumed by the fresh evaluation subprocess.
                with (output / (label + ".log")).open("w") as log:
                    subprocess.run([sys.executable, "-u", "evaluate_rgbd.py", "--config", args.config,
                                    "--weights", str(weights), "--strict-load", "--image-ids", args.panel,
                                    "--diagnostics", "--output", str(result_path)],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                result = json.loads(result_path.read_text())
                assert result["panel_sha256"] == panel_sha and result["validation_images"] == len(records)
                assert all(math.isfinite(result[k]) for k in ("mask_aps", "mask_ap", "mask_ap50", "mask_ap75"))
                result["additional_training_step"] = step
                state["validation_results"].append(result)
                update(status="training", active_evaluation=None)
                print("SCREEN_RESULT", step, result["mask_aps"], result["mask_ap"], flush=True)

            def after_step(self):
                step = self.trainer.iter + 1
                metrics = {k: float(v[0]) for k, v in self.trainer.storage.latest().items()}
                assert all(math.isfinite(v) for v in metrics.values()), "Nonfinite training metric"
                if step == 1:
                    decoder = self.trainer.model.sem_seg_head.predictor
                    gradient = decoder.query_feat.weight.grad
                    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
                    atomic_json(output / "gradient_audit.json", dict(query_gradient_mean_abs=float(gradient.abs().mean()),
                        finite=True, attention_mode=getattr(decoder, "attention_mode", "off"),
                        early_layers=getattr(decoder, "early_layers", 0),
                        query_mode=getattr(decoder, "query_mode", "off")))
                initializer = getattr(self.trainer.model.sem_seg_head.predictor, "query_initializer", None)
                if initializer is not None and (step <= 10 or step % 1000 == 0):
                    gradients = {n: float(p.grad.detach().abs().mean()) if p.grad is not None else None
                                 for n,p in initializer.named_parameters()}
                    assert all(v is None or math.isfinite(v) for v in gradients.values())
                    self.query_inner_gradient_seen |= any(v and v > 0 for n,v in gradients.items() if n.startswith("image."))
                    if step == 10:
                        assert self.query_inner_gradient_seen, "P2 query pooling is disconnected"
                    atomic_json(output / f"query_gradient_step{step}.json", dict(step=step, gradients=gradients,
                        diagnostics={k:float(v) for k,v in initializer.last_diagnostics.items()}))
                if step == 1 or step % 20 == 0 or step == args.max_iter:
                    update(status="training", step=step, metrics=metrics)
                    print("SCREEN_TRAIN_STEP", step, json.dumps(metrics), flush=True)
                if step % args.eval_period == 0 and step < args.max_iter:
                    self.evaluate(self.trainer.checkpointer.get_checkpoint_file(), step)

        monitor = ScreenMonitor()
        trainer.register_hooks([monitor])
        # Measure architecture-only effect separately; don't mistake it for the
        # gain due to training. Control step0 is the frozen reference panel score.
        initial_weights = Path(args.weights).resolve()
        if expected_new:
            initial_weights = output / "model_initialized_inference.pth"
            torch.save({"model": cpu_state_dict(trainer.model)}, initial_weights)
        monitor.evaluate(initial_weights, 0)
        trainer.train()
        final = output / "model_final.pth"
        assert final.exists()
        monitor.evaluate(final, args.max_iter)
        update(status="smoke_test_passed" if args.smoke_test else "completed", step=args.max_iter,
               final_metrics=state["validation_results"][-1], finished_unix=time.time())
    except BaseException as error:
        update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
