# PolyReco — Polymer Recognition & Link Prediction

PolyReco is a research notebook for building graph-based representations of block copolymers and homopolymers, extracting features (Weisfeiler–Lehman kernel, RDKit descriptors, optional ChemBERTa embeddings), and training a two‑stage classifier with automatic tolerance calibration for robust link prediction. It is implemented in Python with PyTorch, DGL, RDKit and scikit‑learn.

> This repository currently centers around the notebook **`PolyReco.ipynb`**. The README explains how to set up the environment, prepare data, reproduce training/evaluation, and run inference on new homopolymers.

---

## ✨ Main features

- Build a DGL graph of the polymer dataset (nodes = monomer@DP; edges store interaction parameters such as `sigma`, `epsilon`).
- Lightweight **WL-subtree kernel** (drop-in replacement for GraKeL) to produce fixed-length graph features.
- Optional **ChemBERTa** text embeddings for monomer strings (`seyonec/ChemBERTa-zinc-base-v1`).
- Per-fold **two-stage** training (feature model → calibrated classifier).
- Automatic tolerance/threshold tuning from validation metrics (balanced accuracy, AUC, AP).
- **Homopolymer link prediction** with weighted ensembling across folds.
- Reproducible metrics saved to CSV files for analysis.

---

## 📁 Repository layout

```
.
├── PolyReco.ipynb          # Main research notebook
└── (your data here)        # e.g., database_25Jun.xlsx
```

---

## 🧾 Input data

The notebook expects an Excel workbook placed at the project root (by default named `database_25Jun.xlsx`) with at least the following sheets/fields:

- **Sheet:** `Homopolymers`
  - **`Big_Smile`** — SMILES or canonical monomer string
  - **`Hard/Soft`** — integer label (e.g., 0/1) used as a node attribute `bip`

- **Sheet:** `BCPs`
  - **`Big_Smile`** — block‑copolymer specification. The notebook parses curly‑brace and comma‑separated forms and extracts monomer blocks and their degree of polymerization (DP).

> ℹ️ If your data uses different column names/formats, update the loader in `DataExtN_kernel()` in the notebook accordingly.

---

## 🧪 Environment setup

**Python:** 3.10+ recommended

We recommend a fresh Conda environment:

```bash
conda create -n polyreco python=3.10 -y
conda activate polyreco

# Core scientific stack
pip install numpy pandas scikit-learn networkx

# PyTorch (choose the right CUDA/CPU build for your system)
# See: https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cpu

# DGL (match your CUDA/CPU)
# See: https://www.dgl.ai/pages/start.html
pip install dgl -f https://data.dgl.ai/wheels/repo.html

# Chemistry & embeddings
pip install rdkit-pypi transformers sentencepiece

# (Optional) Jupyter tooling
pip install jupyter ipykernel
```

> ⚠️ **RDKit** and **DGL** binaries can be platform-specific. If you hit install/import errors, follow their official installation guides for your OS/CUDA version.

---

## ▶️ Quickstart

1. **Place your data** (e.g., `database_25Jun.xlsx`) in the repo root.
2. **Open the notebook** `PolyReco.ipynb` in Jupyter / VS Code.
3. Run the cells in order. Key stages you’ll see:
   - Data loading: builds a DGL graph and per‑monomer artifacts via `DataExtN_kernel()`.
   - Feature engineering: WL‑kernel features, RDKit descriptors, and optional ChemBERTa embeddings.
   - Training: k‑fold pipeline with two‑stage classification and calibration.
   - Evaluation: per‑fold and summary metrics.
   - Inference: homopolymer link prediction and export.

Artifacts/metrics written by the notebook include (among others):
- `per_fold_metrics.csv`, `summary_metrics.csv`
- Sweeps for threshold/tolerance trade‑offs (e.g., `paper_hardAND_grid.csv`, `paper_softANDmin_sweep.csv`, `paper_calibratedAND_sweep.csv`, `paper_lp_filter_tradeoff.csv`)

---

## 📈 Reproduce training & evaluation

The main training orchestration function is:

```python
train_evaluate_kfold_and_collect(...)
```

It handles k‑fold splitting, model fitting, validation‑driven threshold calibration, and exports fold‑level + summary CSVs. Review the **Config** section in the notebook to adjust:
- Feature toggles (WL kernel, RDKit descriptors, ChemBERTa embeddings)
- Lengths/iteration counts for WL features
- Classifier hyperparameters
- Negative sampling / hard negative mining options
- Random seeds for reproducibility

---

## 🔮 Inference: homopolymer link prediction

Use the helper to score candidate links and export results:

```python
# fold_artifacts: list/dict of per‑fold model artifacts produced during training
# fold_weights:   optional list of floats to weight folds (None => uniform)
# LP_FILTER:      optional logit/threshold filter for link prediction

from PolyReco import predict_links_for_homopolymers  # if refactored into a module
# In the notebook, call the function directly.

df_pred = predict_links_for_homopolymers(
    fold_artifacts,
    fold_weights=None,    # or a list like [0.9, 1.1, ...]
    MIN_SUPPORT=0,
    LP_FILTER=0.0
)

df_pred.to_csv("predicted_links.txt", sep="\t", index=False)
print(f"Saved {len(df_pred)} rows to predicted_links.txt")
```

The output is a tab‑separated file with per‑pair scores and any configured metadata. The notebook demonstrates how to compute and save it.

---

## ⚙️ Notes on features

- **WL-subtree kernel:** Implemented in‑notebook as a lightweight replacement for GraKeL; no external GraKeL dependency is required.
- **ChemBERTa embeddings:** Uses `seyonec/ChemBERTa-zinc-base-v1` via Hugging Face `transformers`. Download will occur on first use.
- **Metrics:** Includes ROC‑AUC, AP, F1, balanced accuracy, and helpers like `safe_roc_auc`, `binary_metrics`, and calibration utilities.
- **Reproducibility:** `set_seed(...)` is provided; training code sets seeds for NumPy/PyTorch where applicable.

---

## 📦 Suggested repo structure (if you split the notebook)

If you later refactor into a package:
```
polyreco/
  __init__.py
  data.py              # DataExtN_kernel, loaders/parsers
  features.py          # WL kernel, RDKit, ChemBERTa
  models.py            # Two-stage classifier, calibration
  inference.py         # predict_links_for_homopolymers
  metrics.py           # metrics & threshold search
notebooks/
  PolyReco.ipynb
scripts/
  train.py
  infer.py
```

---

## 🔧 Troubleshooting

- **RDKit import errors:** Prefer conda‑forge or the `rdkit-pypi` wheel; ensure matching Python version.
- **CUDA/CPU mismatches:** Install the correct PyTorch/DGL builds for your CUDA version, or use CPU wheels.
- **Transformers model download issues:** Pre‑download the model or set `HF_HOME` if running in a restricted environment.

---

## 📄 License

Choose a license that matches your intent (e.g., MIT, Apache‑2.0). Add a `LICENSE` file and update this section.

---

## 🙏 Acknowledgements

- **RDKit** for cheminformatics primitives
- **DGL** for graph processing
- **PyTorch** for model training
- **Hugging Face** for ChemBERTa (`seyonec/ChemBERTa-zinc-base-v1`)
