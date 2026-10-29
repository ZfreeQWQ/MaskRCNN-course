#!/usr/bin/env bash
# Use the server's existing driver libraries for this process only.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="/usr/local/iluvatar/lib64:/usr/local/corex/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PROJECT_ROOT/seg_project:$PROJECT_ROOT/seg_project/third_party/msda_shim:$PROJECT_ROOT/seg_project/third_party/Mask2Former:$PROJECT_ROOT/detectron2:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/usr/local/iluvatar/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
# 2026-10-23: this container's NCCL-compatible backend (corex libnccl, see ixccl.h) reliably
# SIGSEGVs the 3rd process (rank 2) in ANY >=3-way NCCL group -- reproduced with a minimal
# torchrun DDP smoke test (tools/check_ddp.py) independent of our model/training code, and
# independent of which physical GPUs are involved (confirmed by testing GPU subsets
# {0,1,2},{1,2,3},{0,1,3}: always crashes at local_rank 2, never a specific GPU). Disabling the
# P2P transport avoids it (isolated via NCCL_P2P_DISABLE=1 on check_ddp.py); this falls back to
# staged host-memory transfers for GPU-to-GPU communication (slower, not faster) rather than
# reinstalling any driver/runtime, per AGENTS.md's "不要通过重装底层环境来尝试修复". Only set when
# not already overridden by the caller.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
exec "$@"
