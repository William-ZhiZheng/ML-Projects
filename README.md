# ML Learning Projects

A monorepo containing various machine learning learning projects and experiments.

## Structure

```
ML-Projects/
├── environment.yml               # Unified conda environment
├── projects/
│   └── reinforcement-learning/    # RL projects and experiments
│       └── sac-line-following/    # SAC policy for line following
├── shared/
│   ├── utils/                    # Common utilities and helpers
│   ├── datasets/                 # Shared datasets
│   └── models/                   # Reusable model architectures
├── docs/                         # Documentation
└── scripts/                      # Utility scripts
```

## Current Projects

### Reinforcement Learning
- **SAC Line Following**: Soft Actor-Critic policy for line following tasks

## Getting Started

1. **Set up the environment:**
   ```bash
   conda env create -f environment.yml
   conda activate ml-projects
   ```

2. **Navigate to your project:**
   ```bash
   cd projects/reinforcement-learning/sac-line-following
   ```

3. **Follow project-specific instructions in each project's README**

The unified environment includes dependencies for:
- Reinforcement Learning (PyTorch, Gymnasium, Stable-Baselines3)
- Computer Vision (OpenCV, Pillow)
- Natural Language Processing (Transformers)
- General ML (scikit-learn, pandas, numpy)
- Visualization (matplotlib, seaborn, plotly)
- Development tools (Jupyter, TensorBoard, Wandb)