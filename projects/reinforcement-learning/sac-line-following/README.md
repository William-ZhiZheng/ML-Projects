# SAC Line Following

Soft Actor-Critic (SAC) policy implementation for line following tasks.

## Overview

This project implements a SAC agent that learns to follow lines in a simulated environment.

## Structure

```
sac-line-following/
├── src/
│   ├── agent/          # SAC agent implementation
│   ├── environment/    # Line following environment
│   ├── models/         # Neural network models
│   └── utils/          # Utilities
├── configs/            # Configuration files
├── experiments/        # Training experiments
└── notebooks/          # Jupyter notebooks for analysis
```

## Getting Started

1. **Set up the unified environment** (from repo root):
   ```bash
   conda env create -f environment.yml
   conda activate ml-projects
   ```

2. **Navigate to this project**:
   ```bash
   cd projects/reinforcement-learning/sac-line-following
   ```

3. **Run training**: `python train.py`
4. **Evaluate**: `python evaluate.py`

Note: All dependencies are managed by the unified conda environment at the repo root.