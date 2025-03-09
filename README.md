# PolyReco

PolyReco is a graph-based neural network project designed for predicting links that satisfy specific criteria. In particular, the project aims to identify links where a measurement (such as strain at break) exceeds a specified threshold. The approach leverages the Deep Graph Library (DGL) and RDKit to construct and process molecular graphs, and uses a GraphSAGE model (built with PyTorch) to predict which links meet the desired criteria. Evaluation is performed using both strict binary classification and a fuzzy (window-based) approach that accounts for real-world measurement variability.

## Overview

- **Data Extraction & Graph Construction:**  
  RDKit converts SMILES strings into molecular graphs, NetworkX builds the initial graphs, and DGL creates final graph structures with node features and edge attributes.

- **Graph Neural Network:**  
  A GraphSAGE model is used to learn node embeddings, and a dot-product predictor computes scores for potential links.

- **Evaluation:**  
  Link predictions are evaluated using:
  - **Strict Classification:** Converts predictions and true link weights into binary classes based on a fixed threshold.
  - **Fuzzy Classification:** Applies an ambiguous window (e.g., ±5 around the threshold) so that predictions near the threshold are counted as correct.

## Dependencies

PolyReco requires the following packages:

- Python 3.6 or higher
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [NetworkX](https://networkx.org/)
- [PyTorch](https://pytorch.org/)
- [DGL (Deep Graph Library)](https://www.dgl.ai/)
- [RDKit](https://www.rdkit.org/)
- [Scikit-Learn](https://scikit-learn.org/)

## Installation

### 1. Create a Conda Environment (Recommended)

Creating a dedicated Conda environment helps manage dependencies without conflicts.

```bash
conda create -n polyreco python=3.8
conda activate polyreco
