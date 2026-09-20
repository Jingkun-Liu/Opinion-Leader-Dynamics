# Opinion Leader Dynamics: How Sparse Attention Shapes Token Clustering

This repository contains implementation for the paper "Opinion Leader Dynamics: How Sparse Attention Shapes Token Clustering". Sparse attention reduces the quadratic cost of global self-attention while retaining strong empirical performance, but how its restricted interactions shape the evolution of token representations remains theoretically underexplored. Modeling tokens as particles on the unit sphere, we introduce **opinion leader dynamics**, a framework that identifies two mechanisms through which token groups converge internally while maintaining distinct limiting directions. In the explicit model, fixed representatives induce a potential that attracts tokens toward distinct local maxima. In the implicit model, disconnected interaction groups evolve toward separate consensus directions. We formulate both models as \emph{reverse Wasserstein gradient flows} and establish exponential convergence under suitable conditions. We further connect these theoretical predictions to token evolution in frontier sparse-attention LLMs that motivate our framework. Across four benchmarks, $\\textcolor{#5178a1}{\\textbf{Kimi-K3}}$, $\\textcolor{#b11f23}{\\textbf{MiniMax-M3}}$,  and $\\textcolor{#5178a1}{\\textbf{DeepSeek-}}\\textcolor{#b11f23}{\\textbf{V4-Flash}}$ consistently exhibit clearer cluster separation and higher clustering scores than the dense-attention model  $\\textcolor{#148e6f}{\\textbf{GLM-4.7-Flash}}$ in projected token representations. These observations support the relevance of the predicted multiple-group structure to trained frontier LLMs, while finite-particle simulations illustrate the theoretical convergence behavior. Together, our results connect restricted token interactions to distinct group-level attractors, providing a dynamical account of how sparse attention can support alignment within groups while preserving separation between them.

## Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/Jingkun-Liu/Opinion-Leader-Dynamics.git](https://github.com/Jingkun-Liu/Opinion-Leader-Dynamics.git)
cd Opinion-Leader-Dynamics
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

## Project Structure

```text
Opinion-Leader-Dynamics/
├── README.md
├── requirements.txt           
├── simulation/
│   ├── main.py                 
│   ├── geometry.py             
│   ├── simulation_core.py      
│   ├── visualization.py        
│   ├── run_simulation.sh      
│   └── explicit_conditional/
│       ├── main.py             
│       ├── geometry.py         
│       ├── simulation_core.py  
│       ├── visualization.py    
│       └── run_simulation.sh   
└── observation/
    ├── main.py                 
    ├── runner.py               
    ├── model_loading.py        
    ├── prompts.py              
    ├── clustering.py           
    ├── baselines.py            
    ├── plotting.py             
    └── run_hellaswag.sh        
```

## Attention Dynamics Simulation

Enter the main simulation directory:

```bash
cd simulation
```

Run any of the three models:

```bash
bash run_simulation.sh standard
bash run_simulation.sh explicit
bash run_simulation.sh implicit
```

## Conditional Explicit Experiment

`simulation/explicit_conditional/` provides a implementation for Explicit Opinion Leader experiments under Assumption 3.1 (geometric separation and dominance conditions).

```bash
cd ./simulation/explicit_conditional
bash run_simulation.sh
```
Results are written to `explicit_conditional/results/`.

## Frontier LLM Analysis

### Model directory requirements

`--config` must point to the configuration file inside the DSV4 inference directory. A compatible layout looks like this:

```text
llm/DS_V4_Flash/
├── inference/
│   ├── config.json
│   ├── model.py
│   └── kernel.py
├── encoding/
│   └── encoding_dsv4.py
└── tokenizer.json
```

The loader adds the parent directory of `config.json` to `sys.path` and then imports:

```python
from model import ModelArgs, Transformer
```

The model-parallel checkpoint directory must contain one shard per process. For a four-way checkpoint:

```text
model0-mp4.safetensors
model1-mp4.safetensors
model2-mp4.safetensors
model3-mp4.safetensors
```

When `--nproc-per-node=4` is used, each rank loads its corresponding `model{rank}-mp4.safetensors` shard.

### Usage

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 \
  --standalone \
  --nproc-per-node=4 \
  main.py eval \
  --ckpt-path .../llm/DS_V4_Flash_mp4 \
  --config .../llm/DS_V4_Flash/inference/config.json \
  --tokenizer-path .../llm/DS_V4_Flash \
  --hellaswag-path .../datasets/hellaswag/data/test-00000-of-00001.parquet \
  --hellaswag-activities all \
  --out-dir results \
  --max-tokens 2048 \
  --max-seq-len 2048 \
  --num-observation-layers 12 \
  --umap-fit-tokens-per-layer 128 \
  --plot-tokens 256 \
  --samples-per-group 100 \
  --umap-components 32 \
  --umap-n-neighbors 10 \
  --umap-min-dist 0.1 \
  --umap-metric cosine \
  --hdbscan-min-cluster-fraction 0.03 \
  --hdbscan-min-samples-fraction 0.01 \
  --hdbscan-cluster-selection-method eom \
  --tilelang-backend auto \
```

## Citation


