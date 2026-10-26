"""Remove zero-area instances only; never train, infer again or tune test scores."""
import argparse
import hashlib
import json
from pathlib import Path

from course.common import atomic_json
from course.submission import repair_empty_masks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", default="/workspace/blender/test_images.json")
    args=parser.parse_args()
    result=repair_empty_masks(args.source,args.metadata,args.output)
    result.update(source=str(Path(args.source).resolve()), output=str(Path(args.output).resolve()),
                  source_sha256=hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
                  output_sha256=hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
                  bytes=Path(args.output).stat().st_size)
    atomic_json(Path(args.output).with_suffix(".repair.json"),result)
    print(json.dumps(result),flush=True)


if __name__=="__main__": main()
