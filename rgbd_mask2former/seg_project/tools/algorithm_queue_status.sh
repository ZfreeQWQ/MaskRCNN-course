set -eu
cd /private/rgzndzy
python - <<'PY'
import datetime,json,sys
from pathlib import Path
sys.path.insert(0,'/private/rgzndzy/seg_project/tools')
from algorithm_queue_worker import process_identity
root=Path('/private/rgzndzy/seg_project/experiments/mask2former')
state=json.loads((root/'algorithm_queue_20261026/state.json').read_text())
result={k:state.get(k) for k in ('status','stage','manager_pid','updated_utc','predecessor','continuation_parent','recommended_model','error')}
result['checked_cst']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
result['manager_alive']=process_identity(state['manager_pid'])==state['manager_start_ticks']
result['queued_plan']=state['plan']
result['jobs']={k:{x:v.get(x) for x in ('status','pid','exit_code','folder')} for k,v in state['jobs'].items()}
result['candidates']={k:{x:v.get(x) for x in ('APs','AP','iteration','output')} for k,v in state['candidates'].items()}
result['latest_decisions']=state['decisions'][-5:]
print(json.dumps(result,indent=2))
PY
