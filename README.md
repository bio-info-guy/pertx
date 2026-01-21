# PERTX: Perturbation Transformer with Graph Priors

**PERTX** is a deep learning framework designed to model single-cell gene expression data with a specific focus on predicting and understanding genetic perturbations. Building upon the architecture of [scGPT](https://github.com/bowang-lab/scGPT), PERTX integrates Graph Neural Networks (GNNs) (like GEARS) to leverage biological prior knowledge (such as Protein-Protein Interaction networks) for robust perturbation embedding and outcome prediction.

## 🧬 Key Features

* **Transformer-Based Foundation**: specialized variations of the scGPT architecture (`BaseModel`, `PertTFGraphModel`) tailored for perturbation tasks.
* **Knowledge-Graph Integration**: Incorporates `LearnableWeightedRGCN` and `LearnableMultiViewGNN` to embed perturbation labels using prior biological knowledge (e.g., STRING DB) rather than simple one-hot encodings.
* **Multi-Task Learning**:
* **Masked Gene Expression Prediction (MLM)**: Reconstructs masked gene expression values.
* **Next-Cell State Prediction**: Predicts the gene expression profile of a cell *after* a specific perturbation.
* **Cell Label Classification**: Predicts the cell label from cell state.
* **Phenotype Score (PS) Prediction**: Regresses continuous phenotype scores (e.g., "Lochness" scores).


* **Advanced Attention Mechanisms**: Supports **Flash Attention** (v2/v3) for efficient training on long sequences.
* **Hybrid Embeddings**: Supports `GeneHybridEmbedding` to selectively freeze or train gene embeddings initialized from pre-trained models like GenePT.

## 📂 Repository Structure

```
pertx/
├── data/               # Data loading and preprocessing
│   ├── dataloader.py   # PertTFDataset and efficient batch collators
│   └── data_util.py    # Utilities for creating train/valid splits
├── hf/                 # HuggingFace-style configuration classes
│   └── config.py       # TransformerModelConfig & PertTFGraphModelConfig
├── layers/             # Neural network building blocks
│   ├── modules.py      # GNNs (RGCN), Decoders (CLS, MVC, PS), & Encoders
│   └── flash_layers.py # Flash Attention implementations
├── models/             # Core model architectures
│   ├── base_model.py   # Base Transformer implementation
│   └── perttf_gnn.py   # Graph-augmented Perturbation Model
├── optim/              # Optimization and Objectives
│   ├── loss.py         # Masked MSE, ZINB, Triplet, & Contrastive losses
│   └── optimizer.py    # Optimizer selection (AdamW, Lion, Prodigy)
├── tokenizer/          # Tokenization logic
│   └── custom_tokenizer.py # Gene-to-token mapping and padding
└── train.py            # Training loop, evaluation, and logging

```

## 🛠️ Installation

Ensure you have a Python environment with PyTorch installed. Key dependencies include:

* **PyTorch** (with CUDA support recommended)
* **Scanpy** & **AnnData** (for single-cell data handling)
* **Torch Geometric** (for GNN layers)
* **Flash Attention** (optional, for speedup)
* **Transformers** (Hugging Face)
* **WandB** (for experiment tracking)

```bash
pip install torch scanpy anndata torch-geometric transformers wandb
# For Flash Attention (requires compatible CUDA):
pip install flash-attn

```

## 🚀 Usage

### 1. Data Preparation

PERTX is designed to work with `AnnData` objects. The data manager handles splitting and tokenization.

```python
from pertx.data.dataloader import PertTFUniDataManager
from pertx.hf.config import PertTFGraphModelConfig

# Load your AnnData
adata = sc.read_h5ad("path/to/data.h5ad")

# Initialize Config
config = PertTFGraphModelConfig(
    ntoken=30000, 
    d_model=512, 
    n_pert=100, # Number of perturbation types
    ...
)

# Prepare Data Manager
manager = PertTFUniDataManager(
    adata, 
    config, 
    expr_layer='X_binned'
)
train_data, train_loader, valid_data, valid_loader, data_info = manager.get_train_valid_loaders()

```

### 2. Model Initialization

Initialize the `PertTFGraphModel`. If you have a prior knowledge graph (e.g., an adjacency matrix of gene interactions), you can pass it to the model.

```python
from pertx.models.perttf_gnn import PertTFGraphModel

model = PertTFGraphModel(
    ntoken=config.ntoken,
    d_model=config.d_model,
    nhead=config.nhead,
    d_hid=config.d_hid,
    nlayers=config.nlayers,
    n_pert=config.n_pert,
    nlayers_pert=3,
    n_ps=1, # Number of phenotype scores to predict
    pert_graph=my_graph_structure, # Optional PyG graph data
    **config.to_dict()
)

```

### 3. Training

Use the wrapper function to start training. This handles the training loop, validation, and logging to Weights & Biases.

```python
from pertx.train import wrapper_train

wrapper_train(
    model=model,
    config=config,
    data_gen=data_info,
    save_dir="./checkpoints"
)

```

## ⚙️ Configuration

The model supports extensive configuration via `PertTFGraphModelConfig`:

* `pred_lochness_next`: Enable prediction of next-state phenotype scores.
* `pert_style`: How to integrate perturbation embeddings (`concat` or AE).
* `use_fast_transformer`: Toggle between `flash` (FlashAttention), `sdpa` (PyTorch SDPA), or `native`.
* `gene_emb_style`: Use `vanilla` learned embeddings or `hybrid` pre-trained embeddings.

## 📊 Outputs & Visualization

The training pipeline automatically logs metrics to **WandB**, including:

* MSE for gene expression reconstruction.
* Classification accuracy for cell types and perturbations.
* UMAP visualizations of embeddings (Cell Type, Perturbation, Predicted States) generated during validation epochs.
