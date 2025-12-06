# CS230 Final Project: Learning a Multi-Head Value Model for Short-Video Recommendation

This repository contains the code for my CS230 (Deep Learning) final project at Stanford:
**“Learning a Multi-Head Value Model for Short-Form Video Recommendation with GNNs.”**

The goal is to build a toy short-video recommender on the **KuaiRec** dataset, predict multiple
engagement signals with a GNN, and then estimate value-model weights over these signals by
matching embeddings of observed and simulated recommendations.

---

## Project overview

Given user–video interactions from KuaiRec, the model predicts four binary engagement heads:

- `y_complete`: whether the video was (almost) completed  
- `y_long`: long-watch event  
- `y_rewatch`: effective rewatch  
- `y_neg`: very short / negative interaction  

I compare:

- Logistic regression on edge features  
- An edge-feature MLP (EdgeMLP)  
- Graph Neural Networks (GNNs) with different hidden dimensions  

All models share the same edge-feature head; the GNN adds message passing over the user–item
bipartite graph. In a second stage, I estimate value-model weights over the four heads by matching
session-level embeddings of observed and simulated recommendation lists.

---

## Repository structure

Rough structure (file names may be slightly different, but the roles are):

- `1. data_preparation.ipynb`  
  - Load raw KuaiRec data and build a clean interaction DataFrame with user, item, and context features.
- `2. data_preparation_GNN.ipynb`  
  - Build the user–item bipartite graph (`edge_index`, `edge_attr`, labels), and train/val/test splits.
- `3. training_GNN.ipynb` / `3. training_GNN.py`  
  - Define logistic, EdgeMLP, and GNN models  
  - Run hyperparameter sweeps (learning rate, batch size) on a tiny subset  
  - Train final models on the full data and compute AUC metrics.
- `4. head_weight_estimation.ipynb`  
  - Use the trained models to simulate recommendations and estimate value-model weights by
    minimizing an embedding-matching loss.

You can read the notebooks in numerical order (`1 → 4`) to follow the full pipeline.

---

## Data

This project uses the **KuaiRec** dataset:

> KuaiRec: A Fully-observed Dataset for Recommender Systems  
> https://kuairec.com/

The dataset is **not** included in this repository.  
To run the code:

1. Go to the KuaiRec website and request/download the data.  
2. Place the raw files under a directory like `data/` (or update paths in the notebooks accordingly).  
3. Adjust any file paths inside the notebooks if your directory layout is different.

---

## Environment and dependencies

Main dependencies (Python):

- `python >= 3.9`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `pytorch`
- `pytorch_geometric` (and its required backends)
- `tqdm`

A simple way to set up an environment:

```bash
conda create -n cs230-gnn python=3.10
conda activate cs230-gnn

pip install numpy pandas scikit-learn matplotlib tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # or cpu version
pip install torch-geometric
