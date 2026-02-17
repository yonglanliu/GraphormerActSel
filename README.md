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

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yonglanliu/GraphormerActSel.git
cd GraphormerActSel
```

```bash
source install.sh
```
