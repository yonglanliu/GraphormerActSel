python3.10 -m venv venv
source venv/bin/activate
pip install -U pip

pip install torch==2.1.* torchvision==0.16.* torchaudio==2.1.* --index-url https://download.pytorch.org/whl/cu118

pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

pip install torch-geometric
pip install torch-sparse

# compiled ops for torch 2.1 + cu118
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html


python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

pip install omegaconf
pip install hydra-core
pip install rdkit-pypi
pip install fastparquet

python install e.
