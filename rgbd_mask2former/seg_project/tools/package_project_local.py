"""Create a verifiable source/experiment-record snapshot, excluding bulk assets."""
import hashlib
import json
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

root = Path('/private/rgzndzy')
output = root/'seg_project/reports/local_sync_20261025'
output.mkdir(parents=True, exist_ok=True)
archive = output/'rgzndzy_source.tar.gz'
manifest_path = output/'source_manifest.json'
if archive.exists() or manifest_path.exists():
    raise SystemExit('Snapshot already exists; do not overwrite')

skip_dirs = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ipynb_checkpoints',
             'build', 'dist', 'node_modules', '.venv'}
skip_suffixes = {'.pyc', '.pyo', '.so', '.o', '.a', '.pth', '.pkl', '.pt', '.onnx', '.zip'}
files, excluded = [], Counter()
for path in sorted(root.rglob('*')):
    relative=path.relative_to(root)
    parts=relative.parts
    if not path.is_file():
        continue
    reason=None
    if any(part in skip_dirs or part.endswith('.egg-info') for part in parts):
        reason='caches_build_and_git_metadata'
    elif parts[0] == 'blender' and relative.as_posix() not in ('blender/README.md', 'blender/submission.csv'):
        reason='raw_dataset'
    elif parts[:2] in (('seg_project','datasets'), ('seg_project','weights'), ('seg_project','backups')):
        reason='datasets_weights_backups'
    elif parts[:3] == ('seg_project','reports','local_sync_20261025'):
        reason='snapshot_outputs'
    elif path.suffix.lower() in skip_suffixes:
        reason='weights_archives_and_compiled_binaries'
    elif 'experiments' in parts:
        if 'snapshots' in parts or path.name.endswith('.pid'):
            reason='redundant_snapshots_or_process_ids'
        elif path.suffix.lower() not in ('.yaml','.yml','.json','.jsonl','.txt','.log','.sh','.md','.csv') and path.name!='last_checkpoint':
            reason='non_source_experiment_asset'
        elif any(word in path.name.lower() for word in ('predictions','coco_instances_results','shard','masks.json','boxes.csv','submission.csv')):
            reason='full_predictions_and_submissions'
    elif path.stat().st_size > 12*1024*1024:
        reason='bulk_non_source_asset'
    if reason:
        excluded[reason]+=1
        continue
    files.append(path)

records=[]
with tarfile.open(archive,'w:gz',compresslevel=5) as stream:
    for path in files:
        relative=path.relative_to(root).as_posix()
        data=path.read_bytes()
        records.append(dict(path=relative, size=len(data), sha256=hashlib.sha256(data).hexdigest()))
        stream.add(path, arcname=relative, recursive=False)
manifest=dict(remote_root=str(root), created_utc=datetime.now(timezone.utc).isoformat(),
              scope='Source, configs, docs, reports and experiment records; bulk data/weights/predictions remain on server.',
              files=records, excluded_counts=dict(excluded), archive_bytes=archive.stat().st_size,
              archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k!='files'},ensure_ascii=False),flush=True)
print('FILES',len(records),'UNCOMPRESSED_BYTES',sum(row['size'] for row in records),flush=True)
