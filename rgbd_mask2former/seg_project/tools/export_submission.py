"""Export existing COCO predictions to the current Kaggle mask-JSON CSV on CPU.

No inference, training, or test annotations are required. The source JSON and old
box CSV are left untouched. Output must be a new file.
"""
import argparse
import csv
import os
from pathlib import Path

from common import load_images, read_json, write_csv, write_json
from validate_submission import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Output already exists; choose a new path to preserve artifacts')
    images = load_images(args.metadata, 'test')
    by_id = {image['id']: image for image in images}
    with open(args.template, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['image_id', 'PredictionString']:
            raise ValueError('Unexpected template columns')
        ids = [int(row['image_id']) for row in reader]
    if len(ids) != len(set(ids)) or set(ids) != set(by_id):
        raise ValueError('Template image ids do not match test metadata')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.partial')
    if temporary.exists():
        parser.error('Partial output already exists; inspect it before retrying')
    try:
        write_csv(temporary, [by_id[image_id] for image_id in ids], read_json(args.predictions))
        report = validate(args.metadata, args.predictions, temporary, manifest=args.manifest)
        # Publish only after every RLE and CSV row has passed validation.
        os.rename(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    report.update(predictions=args.predictions, manifest=args.manifest, template=args.template,
                  output=str(output), template_order_matches=True)
    write_json(str(output) + '.validation.json', report)
    print(report)


if __name__ == '__main__':
    main()
