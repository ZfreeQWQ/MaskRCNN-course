"""Apply the exact two-line scalar-dispatch compatibility fix to pinned MaskDINO."""
import argparse,hashlib
from pathlib import Path

BEFORE='9b45778524e75be7696f1fdb3c41564efae162930d5e8f9f89cff74b98167f1c'
AFTER='0b7c38657c05ca77e335cda87a06499b62b51db40393da3b57551dc7010dfcef'

def main(root):
    path=Path(root)/'maskdino/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu'
    content=path.read_bytes();digest=hashlib.sha256(content).hexdigest()
    if digest==AFTER:
        print('Compatibility fix already applied');return
    assert digest==BEFORE, 'Unexpected upstream file; do not patch an unverified version'
    old=b'AT_DISPATCH_FLOATING_TYPES(value.type(),'
    new=b'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),'
    assert content.count(old)==2
    updated=content.replace(old,new)
    assert hashlib.sha256(updated).hexdigest()==AFTER
    path.write_bytes(updated)
    print('Applied verified scalar-dispatch compatibility fix; rebuild CUDA extension')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('maskdino_root')
    main(p.parse_args().maskdino_root)
