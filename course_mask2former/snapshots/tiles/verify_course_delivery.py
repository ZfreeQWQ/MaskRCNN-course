"""Verify packaged files and a model's isolated source/config without training."""
import argparse
import ast
import json
from pathlib import Path
import sys

from course_batch_predict import verify_bundle
from course_demo import sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--model',choices=['control_30k','boundary_30k'],required=True)
    p.add_argument('--config',action='store_true')
    args=p.parse_args()
    report,config,source,manifest_sha=verify_bundle(args.bundle,args.model)
    manifest=json.loads((args.bundle/'bundle_manifest.json').read_text())
    parsed=0
    for rel in manifest['file_sha256']:
        path=args.bundle/rel
        if path.suffix=='.py':
            ast.parse(path.read_text(),filename=str(path));parsed+=1
    result=dict(status='passed',model=args.model,manifest_sha256=manifest_sha,
                file_count=len(manifest['file_sha256']),python_files_parsed=parsed,
                checkpoint_expected_sha256=report['checkpoint_sha256'],
                checkpoint_loaded=False,training_started=False)
    if args.config:
        sys.path.insert(0,str(source))
        from types import SimpleNamespace
        from mmengine.config import Config
        from mmengine.utils import import_modules_from_strings
        from run_course_experiment import build_config
        saved=Config.fromfile(str(config))
        rebuilt=build_config(SimpleNamespace(**report['args']),
             [dict(id=1,name='component',supercategory='component')])
        for field in ('model','train_cfg','param_scheduler','optim_wrapper'):
            a=saved[field].to_dict() if hasattr(saved[field],'to_dict') else saved[field]
            b=rebuilt[field].to_dict() if hasattr(rebuilt[field],'to_dict') else rebuilt[field]
            if a!=b:raise ValueError('Rebuilt configuration mismatch: '+field)
        import_modules_from_strings(**saved.custom_imports)
        plan=json.loads((args.bundle/'models'/args.model/'plan.json').read_text())
        origins={}
        for rel in plan['source_sha256']:
            if not rel.endswith('.py'):continue
            name=rel[:-3].replace('/','.')
            if name.endswith('.__init__'):name=name[:-9]
            module=sys.modules.get(name)
            if module is not None and getattr(module,'__file__',None):
                actual=Path(module.__file__).resolve()
                if actual!=(source/rel).resolve():
                    raise ValueError('Source leaked outside snapshot: '+str(actual))
                origins[name]=str(actual.relative_to(args.bundle.resolve()))
        result.update(config_equivalent_fields=['model','train_cfg','param_scheduler','optim_wrapper'],
                      imported_project_module_origins=origins,
                      official_data_opened=False)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
    main()
