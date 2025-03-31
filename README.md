# PNMLR: Personalized Route Recommendations with Graph Attention Networks
[![IEEE Access](https://img.shields.io/badge/IEEE%20Access-Published-blue)](https://doi.org/10.1109/ACCESS.2025.3555049)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org/)

<p align="center">
  <img src="Cover_image.webp" alt="Cover image" width="500"/>
</p>

## Overview

PNMLR (Personalized Neuro-MLR) enhances route recommendations by embedding user-specific preferences into the prediction process. This repository contains the implementation from our [IEEE Access paper](https://doi.org/10.1109/ACCESS.2025.3555049).

## Installation

```bash
# Clone the repository
git clone https://github.com/ludocomito/GNN-Route-Recommendation.git
cd GNN-Route-Recommendation

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

The model uses the [GeoLife GPS dataset](https://www.microsoft.com/en-us/research/project/geolife-building-social-networks-using-human-location-history/). The complete preprocessing pipeline is in `dataset_preprocessing.ipynb`.

To prepare the dataset:
1. Download the GeoLife GPS dataset
2. Run all cells in `dataset_preprocessing.ipynb`
3. Verify the processed data is in the `dataset/` directory

## Running the Model

### Training

```bash
# Train with Graph Attention Networks and preference concatenation
python main.py -gnn "GAT" -merging_strategy "cat" -num_pref_layers 3 -run_name "pnmlr_gat"

# Train with Graph Convolutional Networks and preference averaging
python main.py -gnn "GCN" -merging_strategy "avg" -num_pref_layers 1 -run_name "pnmlr_gcn"
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-gnn` | Graph neural network type ("GAT" or "GCN") | "GAT" |
| `-merging_strategy` | Preference embedding strategy ("cat" or "avg") | "cat" |
| `-num_pref_layers` | Number of layers in preference MLP | 1 |
| `-run_name` | Experiment name | "default_run" |
| `-num_epochs` | Training epochs | 100 |
| `-batch_size` | Batch size | 32 |
| `-optimizer` | Optimizer type | "adam" |
| `-embedding_size` | Node embedding size | 128 |
| `-hidden_size` | Hidden layer size | 256 |
| `-eval_freq` | Evaluation frequency (epochs) | 1 |

### Monitoring Training

The model automatically logs metrics to the console. If WandB is enabled (`-disable_debug_mode False`), metrics will also be logged to WandB for visualization.

### Loading a Trained Model

```python
import torch
from constants import CHECKPOINT_PATH

# Load saved model
model = torch.load(CHECKPOINT_PATH + "pnmlr_gat.pt")
```

## Model Evaluation

Evaluation metrics are automatically calculated at the end of training:

- Precision: Fraction of correctly predicted route segments
- Recall: Fraction of actual route segments correctly predicted
- F1 Score: Harmonic mean of precision and recall
- Reachability: Percentage of routes that reach the correct destination

Results are saved to `checkpoints/model_stats/[run_name].txt`.

## Project Structure

```
├── main.py                 # Main training script
├── model.py                # PNMLR model implementation
├── modules.py              # Model components (GNN, MLP, etc.)
├── args.py                 # Command-line arguments
├── constants.py            # File paths and constants
├── dataloading_utils.py    # Data loading utilities
├── dataset_preprocessing.ipynb  # Dataset preprocessing pipeline
├── requirements.txt        # Dependencies
└── checkpoints/            # Saved models and results
```

## Citation

If you use this code, please cite our paper:

```bibtex
@article{ponzi2025pnmlr,
  title={PNMLR: Enhancing Route Recommendations with Personalized Preferences Using Graph Attention Networks},
  author={Ponzi, Valerio and Comito, Ludovico and Napoli, Christian},
  journal={IEEE Access},
  volume={11},
  year={2025},
  doi={10.1109/ACCESS.2025.3555049},
  publisher={IEEE}
}
```

## Authors

- Valerio Ponzi - Sapienza University of Rome
- [Ludovico Comito](https://github.com/ludocomito) - Sapienza University of Rome
- Christian Napoli - Sapienza University of Rome
