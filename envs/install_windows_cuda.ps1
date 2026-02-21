$ErrorActionPreference = "Stop"

python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"

python -m pip install -U pip

# DGL CUDA wheels (Windows support varies by release; this is the official wheel repo for cu118)
python -m pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

# PyG compiled ops for torch 2.1.0 + cu118
python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv `
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

python -m pip install torch-geometric

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
python -m pip install -e .
