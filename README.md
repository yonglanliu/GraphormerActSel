# GraphormerActSel

GraphormerActSel is a deep learning framework for joint molecular **activity and selectivity prediction**, built upon a pretrained **Graphormer** backbone.

The model leverages graph transformer representations of molecular structures and fine-tunes them for downstream bioactivity and selectivity modeling tasks.

---

## Overview

Accurate prediction of both activity and selectivity is critical in drug discovery.  
GraphormerActSel integrates:

- Graph-based molecular representation learning
- Transformer architecture (Graphormer backbone)
- Multi-task prediction for activity and selectivity
- Transfer learning from pretrained graph models

This repository contains code for:

- Model fine-tuning
- Training pipelines
- Evaluation scripts
- Reproducibility of reported results

---

## Model Architecture

GraphormerActSel is built on:

- **Graphormer** (Ying et al., 2021)
- Pretrained molecular graph representations
- Task-specific prediction heads for:
  - Activity prediction
  - Selectivity prediction

The model supports multi-task and single-task training modes.

---

## Set up

### 1. Clone the repository

```bash
git clone https://github.com/yonglanliu/GraphormerActSel.git
cd GraphormerActSel
```

---

### 2. Installation

#### 1) macOS —- envs/environment.macos.yml

```bash
conda env create -f envs/environment.macos.yml
conda activate graphormerActSel
pip install -e .
python -c "import torch; print(torch.__version__); print('mps available:', hasattr(torch.backends,'mps') and torch.backends.mps.is_available())"
```

#### 2) Linux CUDA -- envs/install_linux_cuda.sh

conda installs CUDA-enabled PyTorch, then pip installs PyG/DGL wheels matching that torch+cuda.

```bash
conda env create -f envs/environment.linux.cu118.yml
conda activate graphormerActSel
bash envs/install_linux_cuda.sh
```

#### 3) Pure pip option (Linux CUDA only) -- envs/requirements.linux.cu118.txt

```bash
python3.10 -m venv graphormerActSel
source graphormerActSel/bin/activate
pip install -U pip
pip install -r requirements.linux.cu118.txt
pip install -e .
```

#### 4) Windows 

**Note: Quick check if there is GPU**

```bash
nvidia-smi
```

If there is GPU, use CUDA installation, otherwise use CPU installation.

#### NVIDIA GPU (CUDA 11.8) —- envs/environment.windows.cu118.yml

```bash
conda env create -f envs/environment.windows.cu118.yml
conda activate graphormerActSet
.\envs\install_windows_cuda.ps1
```

#### Windows CPU-only — envs/environment.windows.cpu.yml

```bash
conda env create -f envs/environment.windows.cpu.yml
conda activate graphormerActSel
pip install -e .
```
---

## Acknowledgment

This project builds upon the Graphormer implementation by Microsoft Research.

Original repository:
[https://github.com/microsoft/Graphormer](https://github.com/microsoft/Graphormer)

It incorporates code from:

- Graphormer (Microsoft Research)
- fairseq (Meta AI)

Both licensed under the MIT License.

---

## Modifications

This repository extends the original Graphormer implementation with the following enhancements:

* Multi-task learning framework for simultaneous prediction of compound activity and isoform selectivity across protein families
* Hybrid objective function combining regression loss with pairwise selectivity ranking loss
* Explicit shortest-path computation between atom pairs for improved structural encoding
* Refactoring to native PyTorch modules, replacing framework-specific components for improved flexibility and maintainability
* Adapter-based soft parameter sharing to enable efficient multi-target modeling within the same backbone
* Customized training pipeline, including multi-task scheduling and evaluation strategies
* Replacement of the quant-noise implementation with the original fairseq (Meta/Facebook) version for consistency and stability
* Custom data processing pipeline tailored for structure-aware molecular modeling

---
