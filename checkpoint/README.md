## Download Graphormer's Weights

To download Graphormer weights, you can use the initialization script found in
<a href="https://github.com/microsoft/Graphormer/blob/main/graphormer/pretrain/init.py">__init__.py</a>. 
Alternatively, you can download the pre-trained models directly using <code>wget</code> or <code>curl</code>.

### 1. Using the Python API
If you have the package installed, you can trigger the download programmatically:

```bash
from graphormer.pretrain import load_pretrained_model

# This will automatically download and cache the weights
model = load_pretrained_model("graphormer-base-pcqm4mv1")
```

### 2. Manual Download (Direct Links)

For manual installation, use the following commands to download the checkpoints. We recommend placing them in a ./checkpoints directory.

| Model | Variant |	Source | Command|
|---|---|---|---|
| Graphormer-Base |	PCQM4M-LSC	| wget | https://graphormer.blob.core.windows.net/lp-models/graphormer_base_pcqm4m_v2.pt |
| Graphormer-Slim	| PCQM4M-LSC	| curl | -O https://graphormer.blob.core.windows.net/lp-models/graphormer_slim_pcqm4m_v2.pt |

