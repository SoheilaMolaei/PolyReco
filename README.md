# PolyReco

PolyReco is a graph-based neural network project designed for predicting links that satisfy specific criteria. In this project, links (edges) are predicted based on whether their associated measurements (for example, strain at break) exceed a defined threshold. The model uses the Deep Graph Library (DGL) and RDKit to construct and process molecular graphs, and a GraphSAGE model (built with PyTorch) to perform the link predictions.


### 1. Install PyTorch

Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) for installation instructions. For a CPU-only version, run:

```bash
pip install torch torchvision torchaudio
```

### 2. Install DGL

For the CPU-only version, install DGL via Conda:

```bash
conda install -c dglteam dgl-cpu
```

For GPU support, install with:

```bash
conda install -c dglteam dgl
```

### 3. Install RDKit

Install RDKit using Conda:

```bash
conda install -c rdkit rdkit
```

### 4. Install Other Dependencies

Install the remaining Python packages:

```bash
pip install numpy pandas networkx scikit-learn
```


### What Happens When You Run the Script?

- Extracts data from Excel sheets.
- Constructs molecular graphs using RDKit, NetworkX, and DGL.
- Trains the model.
- Evaluates link predictions.
- Saves predicted edges and evaluation results for later inspection.

## Troubleshooting

- **Installation Issues:**  
  If you have difficulties installing RDKit or DGL, please refer to their official documentation:
  - [RDKit Documentation](https://www.rdkit.org/docs/)
  - [DGL Documentation](https://docs.dgl.ai/)

- **Dependency Conflicts:**  
  Running the project in a dedicated Conda environment should help avoid conflicts between package versions.

