"""One held-out scene panel, one final-budget winner vs matched control; no training."""
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import zipfile
import torch
from course.common import atomic_json
from course.confirmation import select_winner, validate_panels


def main():
    root = Path(__file__).resolve().parent
    suite = root / "runs/screen_v2_confirmation"
    suite.mkdir(parents=True, exist_ok=True)
    lock = (suite/"queue.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (suite/"suite.json").exists(): raise RuntimeError("Confirmation already exists: never repeat or overwrite")
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).strip()
    previous = json.loads((root/"runs/screen_v2/suite.json").read_text())
    assert previous["status"] == "completed"
    winner = select_winner(previous["jobs"],previous["control_metrics"])
    assert winner is not None, "No consistent final-budget gain: don't use confirmation"
    screen = json.loads((root/"runs/screen_v1/panel_screen.json").read_text())
    panel = json.loads((root/"runs/screen_v1/panel_confirmation.json").read_text())
    validate_panels(screen,panel)
    atomic_json(suite/"panel.json",panel)
    panel_sha = hashlib.sha256((suite/"panel.json").read_bytes()).hexdigest()
    gpu_memory = subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
    assert all(int(v.strip())<256 for v in gpu_memory.splitlines()[:2]), "GPUs occupied: never stop unrelated jobs"
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    archive = subprocess.check_output(["git","archive","--format=zip",revision],cwd=root)
    source = suite/"source";source.mkdir()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert all((source/n).resolve().is_relative_to(source) for n in z.namelist())
        z.extractall(source)
    for n in ("weights","data_cache"): (source/n).symlink_to(root/n,target_is_directory=True)
    atomic_json(source/".source_revision.json",dict(commit=revision,dirty=False,immutable_snapshot=True,
                archive_sha256=hashlib.sha256(archive).hexdigest()))
    jobs={"i00_control":dict(gpu="0",config="configs/screen/i00_control.yaml",
                            training_checkpoint=str(root/"runs/screen_v1/i00_control/model_final.pth"),status="pending"),
          winner:dict(gpu="1",config=f"configs/screen/{winner}.yaml",
                      training_checkpoint=str(root/"runs/screen_v2"/winner/"model_final.pth"),status="pending")}
    state=dict(status="preparing",commit=revision,source=str(source),controller_pid=os.getpid(),jobs=jobs,
               started_unix=time.time(),winner=winner,panel_sha256=panel_sha,
               selection="Best final3000 APs among positive candidates with >=2 positive trained checkpoints; selected BEFORE confirmation",
               policy="One final-checkpoint winner vs same-budget control on pre-frozen200 images /100 disjoint scenes; no training/full-val/test/TTA; no fallback candidate search",
               screen_final_aps_delta=previous["paired_final_aps_deltas"][winner],test_used=False)
    processes,logs={},[]
    def update(**values):
        state.update(values);state["updated_unix"]=time.time()
        atomic_json(suite/"suite.json",state)
    def interrupted(signum,frame):raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        update()
        for name,job in jobs.items():
            path=Path(job["training_checkpoint"])
            original=torch.load(path,map_location="cpu",weights_only=False)["model"]
            assert hasattr(original,"_metadata") and original._metadata["sem_seg_head"]["version"]==2
            export=suite/(name+"_inference.pth")
            torch.save({"model":original},export)
            reloaded=torch.load(export,map_location="cpu",weights_only=False)["model"]
            assert original._metadata==reloaded._metadata
            assert list(original)==list(reloaded) and all(torch.equal(v,reloaded[k]) for k,v in original.items())
            job.update(weights=str(export),weights_sha256=hashlib.sha256(export.read_bytes()).hexdigest(),
                       training_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       checked_tensors=len(original),weight_transfer_exact=True)
            del original,reloaded
        update(status="running")
        for name,job in jobs.items():
            log=(suite/(name+".log")).open("a");logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=job["gpu"],OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
            p=subprocess.Popen([sys.executable,"-u","evaluate_rgbd.py","--config",job["config"],
                "--weights",job["weights"],"--strict-load","--image-ids",str(suite/"panel.json"),
                "--diagnostics","--output",str(suite/(name+".json"))],cwd=source,env=env,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
            processes[name]=p;job.update(status="running",pid=p.pid)
            update()
        while True:
            for name,p in processes.items():
                if p.poll() is None:continue
                if p.returncode:raise RuntimeError(f"{name} failed: {p.returncode}")
                if jobs[name]["status"]!="completed":
                    result=json.loads((suite/(name+".json")).read_text())
                    assert result["validation_images"]==200 and result["panel_sha256"]==panel_sha
                    jobs[name].update(status="completed",metrics=result)
            update()
            if all(j["status"]=="completed" for j in jobs.values()):
                a,b=jobs["i00_control"]["metrics"],jobs[winner]["metrics"]
                delta=b["mask_aps"]-a["mask_aps"]
                update(status="completed",finished_unix=time.time(),confirmation_aps_delta=delta,
                       directional_gain_repeated=delta>0,
                       conclusion="Direction repeats on one independent small panel, not significance/full-val proof" if delta>0 else
                                  "Screen gain not repeated: do not promote and do not search other candidates on this panel")
                break
            time.sleep(10)
    except BaseException as error:
        for p in processes.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in processes.values():
            if p.poll() is None:
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        update(status="stopped" if isinstance(error,KeyboardInterrupt) else "failed",error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        for log in logs:log.close()


if __name__=="__main__":main()
