#!/usr/bin/env bash

# Ensure env is active
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"

# 1) DGL CUDA 11.8 wheels
# If you need a specific DGL version, pin it (e.g., dgl==2.0.0) based on your project.
pip install -U pip
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

# 2) PyG compiled ops matching torch 2.1 + cu118
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# 3) PyG meta-package
pip install torch-geometric

# Sanity checks
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

import dgl
print("dgl:", dgl.__version__)

import torch_geometric
print("pyg:", torch_geometric.__version__)
PY

# Install your repo
pip install -e .
