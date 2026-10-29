"""Fetch only the official R50 instance checkpoint; never download COCO itself."""
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

import yaml

ROOT = Path('/workspace')
MODEL = 'mask2former_r50_8xb2-lsj-50e_coco'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    repo = ROOT / 'mmdetection'
    metadata = yaml.safe_load((repo / 'configs/mask2former/metafile.yml').read_text())
    entry = next(m for m in metadata['Models'] if m['Name'] == MODEL)
    url = entry['Weights']
    checkpoint = ROOT / 'checkpoints' / url.rsplit('/', 1)[-1]
    expected_prefix = checkpoint.stem.rsplit('-', 1)[-1]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        partial = checkpoint.with_suffix('.pth.partial')
        last_print = time.monotonic()
        print(f'Downloading official checkpoint: {url}', flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as out:
            total = int(response.headers.get('Content-Length', 0))
            copied = 0
            while True:
                data = response.read(4 * 1024 * 1024)
                if not data:
                    break
                out.write(data)
                copied += len(data)
                if time.monotonic() - last_print > 10:
                    print(f'Downloaded {copied / 1e6:.1f}/{total / 1e6:.1f} MB', flush=True)
                    last_print = time.monotonic()
        actual = digest(partial)
        if not actual.startswith(expected_prefix):
            raise RuntimeError(f'Checkpoint SHA256 mismatch: {actual} vs {expected_prefix}')
        os.replace(partial, checkpoint)
    actual = digest(checkpoint)
    if not actual.startswith(expected_prefix):
        raise RuntimeError(f'Existing checkpoint SHA256 mismatch: {actual}')
    manifest = dict(
        model=MODEL, source=url, checkpoint=str(checkpoint), sha256=actual,
        config=str(repo / entry['Config']),
        image=str(repo / 'demo/demo.jpg'),
        note='COCO-pretrained RGB demo; not the course RGB-D dataset.',
    )
    (ROOT / 'checkpoints/smoke_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
