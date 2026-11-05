"""External paths only; no dataset, checkpoint or host credentials in source."""
import os,sys
from pathlib import Path
WORKDIR=Path(os.environ.get('CCC_WORKDIR','./outputs')).resolve()
DATA_ROOT=Path(os.environ.get('CCC_DATA_ROOT','./blender')).resolve()
WEIGHTS=Path(os.environ.get('CCC_WEIGHTS','./weights')).resolve()
POSTPROCESS_WEIGHTS=Path(os.environ.get('CCC_POSTPROCESS_WEIGHTS',str(Path(__file__).parent/'artifacts'))).resolve()
MASKDINO_SRC=Path(os.environ.get('CCC_MASKDINO_SRC','./MaskDINO')).resolve()
GEOMETRY_STATS=Path(os.environ.get('CCC_GEOMETRY_STATS','./geometry_stats.json')).resolve()
IMAGE_METADATA=Path(os.environ.get('CCC_IMAGE_METADATA','./images.json')).resolve()
sys.path[:0]=[str(MASKDINO_SRC/'maskdino/modeling/pixel_decoder/ops'),str(MASKDINO_SRC)]+[p for p in os.environ.get('CCC_EXTRA_PYTHONPATH','').split(os.pathsep) if p]
UPSTREAM_COMMIT='3831d8514a3728535ace8d4ecc7d28044c42dd14'
