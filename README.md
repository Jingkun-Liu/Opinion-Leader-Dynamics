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

If `config.json` and `model.py` are not in the same inference directory, the run will fail with `ModuleNotFoundError: No module named 'model'`.

The model-parallel checkpoint directory must contain one shard per process. For a four-way checkpoint:

```text
model0-mp4.safetensors
model1-mp4.safetensors
model2-mp4.safetensors
model3-mp4.safetensors
```

When `--nproc-per-node=4` is used, each rank loads its corresponding `model{rank}-mp4.safetensors` shard.

### Self-test

```bash
cd github/observation
python main.py self-test --seed 0
```

The self-test builds synthetic hidden states with three known groups and checks that HDBSCAN recovers three clusters, final labels are reused without reclustering, snapshot labels remain fixed, and the final-layer cosine silhouette exceeds the early-layer value.

### HellaSwag example

A four-GPU background launcher is provided:

```bash
cd github/observation
bash run_hellaswag.sh
```

The current launcher is configured for:

- `CUDA_VISIBLE_DEVICES=4,5,6,7`;
- `torchrun --nproc-per-node=4`;
- machine-specific absolute model, checkpoint, and HellaSwag paths;
- at most 100 samples per activity group;
- a 32-dimensional clustering UMAP and 12 observed layers;
- background execution with stdout and stderr redirected to the result directory.

Update the GPU IDs and absolute paths before running the script on another machine.

An equivalent foreground command, launched from `github/observation`, is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc-per-node=4 \
  main.py eval \
  --ckpt-path ../../llm/DS_V4_Flash_mp4 \
  --config ../../llm/DS_V4_Flash/inference/config.json \
  --tokenizer-path ../../llm/DS_V4_Flash \
  --hellaswag-path ../../datasets/hellaswag/data/test-00000-of-00001.parquet \
  --out-dir ./results_dsv4_hellaswag \
  --max-tokens 2048 \
  --max-seq-len 2048 \
  --num-observation-layers 12 \
  --umap-components 32 \
  --tilelang-backend auto
```

### Input formats

Each `eval` run accepts exactly one input source:

- `--hellaswag-path FILE.parquet`: expects `ctx` and four `endings`; it also reads metadata such as `label`, `activity_label`, and `source_id` when present.
- `--prompt-file FILE`: treats one UTF-8 text file as a single sample.
- `--prompt-jsonl FILE`: each nonempty line must contain at least `{"text": "..."}` and may include `tag`/`id` and `group`.
- `--prompt-dir DIR`: recursively reads `.txt` and `.md` files; the first subdirectory is used as the group name.

Example JSONL input:

```json
{"tag":"sample_001","group":"reasoning","text":"Explain why the sky appears blue."}
{"tag":"sample_002","group":"coding","text":"Write a binary search implementation."}
```

### Sampling and token limits

| Option | Description |
| --- | --- |
| `--hellaswag-activities` | Comma-separated activity labels, or `all` |
| `--samples-per-group` | Maximum samples per group; `0` disables the limit |
| `--sample-seed` | Seed for grouped sampling, UMAP, and the Gaussian baseline |
| `--max-tokens` / `--max-seq-len` | The capture limit is the smaller of these two values |
| `--long-prompt-policy error` | Fail when a prompt exceeds the capture limit |
| `--long-prompt-policy head_tail` | Preserve tokens from both the beginning and end |
| `--truncation-head-fraction` | Fraction assigned to the head under `head_tail`; internally clamped to `[0.1, 0.9]` |
| `--resume` | Reuse an existing JSON only when its analysis configuration matches exactly |

### UMAP and HDBSCAN options

| Option | Description |
| --- | --- |
| `--spherical-cluster-space umap` | Cluster in shared UMAP coordinates |
| `--spherical-cluster-space hidden` | Cluster in the full L2-normalized hidden space |
| `--umap-components` | Number of clustering UMAP dimensions; sphere visualization uses a separate 3D map |
| `--umap-fit-tokens-per-layer` | Maximum tokens per layer used to fit the shared UMAP |
| `--umap-n-neighbors` / `--umap-min-dist` | UMAP neighborhood and compactness parameters |
| `--hdbscan-min-cluster-size` | Absolute minimum cluster size; `0` derives it from a fraction with a floor of 10 |
| `--hdbscan-min-samples` | Absolute `min_samples`; `0` derives it from a fraction with a floor of 5 |
| `--hdbscan-cluster-selection-method` | `eom` or `leaf` |
| `--assign-all-tokens` | Assign HDBSCAN noise points to the nearest cluster; enabled by default |
| `--plot-tokens` | Maximum number of tokens displayed in each sphere snapshot |

When `--assign-all-tokens` is disabled, noise retains label `-1` and is excluded from the cosine silhouette calculation.

### TileLang backend selection

`--tilelang-backend` accepts:

- `auto`: uses `tvm_ffi` when a complete CUDA Toolkit containing both `nvcc` and `cuda_runtime.h` is found; otherwise uses `nvrtc`.
- `tvm_ffi`: requires a complete CUDA Toolkit and reports all searched paths when none is found.
- `nvrtc`: compiles through the CUDA runtime compiler and can be used when the system does not provide `nvcc`.

The loader derives `TILELANG_TARGET` from the active GPU compute capability. Its compilation cache is controlled by `DSV4_TILELANG_CACHE_DIR`; when unset, an architecture-specific directory under the active environment is used.

### Observation outputs

Each run writes the following files under `--out-dir`:

```text
sample_manifest.json                  # Inputs, groups, and analysis configuration
gaussian_baseline_hidden_umap.json    # Gaussian initialization baseline
<tag>_hidden_umap.json                # Token data, clustering, and layer-wise metrics
<tag>_hidden_umap.png                 # Sphere snapshots and silhouette curve
hidden_umap_summary.json              # Compact summary of the complete run
```

Each sample JSON includes final-layer HDBSCAN labels, membership probabilities, outlier scores, noise fraction, cluster persistence, layer-wise fixed-label cosine silhouettes, token text, and sphere snapshot coordinates.

## Troubleshooting

### `ModuleNotFoundError: No module named 'model'`

The `--config` path is usually incorrect. Confirm that it points to the real `DS_V4_Flash/inference/config.json` and that `model.py` exists in the same directory. Remember that relative paths are resolved after any `cd` performed by a launcher script.

### Missing checkpoint shard

The number of `torchrun` processes must match the model-parallel degree used during checkpoint conversion. For example, an `mp4` checkpoint requires four ranks and all four `model{rank}-mp4.safetensors` files.

### UMAP reports `cannot cache function ... no locator available`

This indicates an installation or cache compatibility problem involving `umap-learn`, `pynndescent`, and Numba rather than invalid clustering input. Reinstall mutually compatible versions in a clean environment, make sure the package `.py` sources are present, and verify that the cache directory is writable.

### CUDA out of memory

Reduce the following options first:

- `--max-tokens` and `--max-seq-len`;
- `--umap-fit-tokens-per-layer`;
- `--plot-tokens`;
- the number of evaluated samples.

Reducing `--plot-tokens` only reduces visualization size; it does not significantly reduce memory used during the model forward pass.

### Non-finite simulation state

Reduce `--dt` or `--beta`. The Explicit model also enforces `beta * max(leader radius) < 80` to prevent float32 exponential overflow.

## Reproducibility

- Simulation initialization is controlled by `--seed`.
- Observation sampling, UMAP, and the Gaussian baseline are controlled by `--sample-seed`.
- GPU, CUDA, TileLang, UMAP/Numba, and HDBSCAN versions can still introduce small numerical differences.
- Record the complete command, package versions, `sample_manifest.json`, and `hidden_umap_summary.json` for reproducible runs.

## Development checks

Compile all Python sources:

```bash
python -m compileall -q github/simulation github/observation
```

Inspect the three command-line interfaces:

```bash
python github/simulation/main.py --help
python github/simulation/explicit_conditional/main.py --help
python github/observation/main.py --help
```
