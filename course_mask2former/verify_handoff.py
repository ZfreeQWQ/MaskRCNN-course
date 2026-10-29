"""Read-only source/evidence verification; never imports archived production code."""
import ast
import hashlib
import json
from pathlib import Path


def verify():
    root = Path(__file__).resolve().parent
    source = json.loads((root / 'source_manifest.json').read_text())
    evidence = json.loads((root / 'evidence_manifest.json').read_text())
    assert len(source['files']) == source['count'] == 538
    manifest = json.loads((root / 'handoff_manifest.json').read_text())
    expected = {x['path'] for x in manifest['files']}
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.name != 'handoff_manifest.json'}
    assert actual == expected, {'missing': sorted(expected - actual),
                                'unexpected': sorted(actual - expected)}
    for row in source['files'] + evidence['files'] + manifest['files']:
        relative = Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        p = root / relative
        assert p.is_file() and not p.is_symlink(), row['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256'], row['path']
    parsed = {'python': 0, 'json': 0}
    for name in expected:
        p = root / name
        if p.suffix == '.py':
            ast.parse(p.read_text(), filename=name)
            parsed['python'] += 1
        elif p.suffix == '.json':
            json.loads(p.read_text())
            parsed['json'] += 1
        assert p.suffix.lower() not in {'.pth', '.pt', '.csv', '.png', '.jpg', '.zip', '.tar', '.gz', '.log'}, name
    results = json.loads((root / 'results.json').read_text())
    assert results['best'] == 'g_nineview'
    assert abs(results['metrics']['g_nineview']['APs'] - 56.68493022152318) < 1e-10
    assert results['current_experiment_queue'] == 'closed'
    print(json.dumps(dict(status='passed', frozen_sources=source['count'],
                          copied_evidence=len(evidence['files']),
                          package_files=len(expected) + 1, parsed=parsed,
                          GPU_used=False, model_imported=False,
                          inference_or_AP_recomputed=False)))


if __name__ == '__main__':
    verify()
