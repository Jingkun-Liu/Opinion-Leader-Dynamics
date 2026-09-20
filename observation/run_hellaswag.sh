set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR=./results_dsv4_hellaswag_riemann_umap32_spherical_hdbscan
LOG_FILE="${OUT_DIR}/run_hellaswag_riemann_umap.log"
mkdir -p "${OUT_DIR}"

nohup env \
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 \
  torchrun \
  --standalone \
  --nproc-per-node=4 \
  main.py eval \
  --ckpt-path /nvme/jkliu/nsa_iclr/llm/DS_V4_Flash_mp4 \
  --config /nvme/jkliu/nsa_iclr/llm/DS_V4_Flash/inference/config.json \
  --tokenizer-path /nvme/jkliu/nsa_iclr/llm/DS_V4_Flash \
  --hellaswag-path /nvme/jkliu/nsa_iclr/datasets/hellaswag/data/test-00000-of-00001.parquet \
  --hellaswag-activities all \
  --out-dir "${OUT_DIR}" \
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
  > "${LOG_FILE}" 2>&1 &

echo "started HellaSwag Riemann UMAP pid=$! log=${LOG_FILE}"
