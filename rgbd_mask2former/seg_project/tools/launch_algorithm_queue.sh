set -eu
cd /private/rgzndzy
python - <<'PY'
import json, subprocess
from pathlib import Path
root=Path('/private/rgzndzy/seg_project')
out=root/'experiments/mask2former/algorithm_queue_20261026'
out.mkdir(exist_ok=True)
preflight=subprocess.run([str(root/'run.sh'),'python',str(root/'tools/run_algorithm_queue.py'),'--preflight'],
                         capture_output=True,text=True,check=True)
report=json.loads(preflight.stdout)
(out/'preflight.json').write_text(json.dumps(report,indent=2))
log=out/'manager.log'
with log.open('a') as stream:
 proc=subprocess.Popen([str(root/'run.sh'),'python',str(root/'tools/run_algorithm_queue.py')],
     cwd=root.parent,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
     start_new_session=True,close_fds=True)
(out/'launch.pid').write_text(str(proc.pid)+'\n')
print(json.dumps({'launched_manager_pid':proc.pid,'log':str(log),'preflight':report},indent=2))
PY
