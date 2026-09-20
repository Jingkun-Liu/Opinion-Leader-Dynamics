set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/main.py" \
  --n 180 \
  --dim 3 \
  --beta 1.0 \
  --T 0.05 \
  --dt 0.01 \
  --seed 42 \
  --device auto \
  --leaders 6 \
  --leader-radii auto \
  --leader-weights 1.01,0.95,1.02,0.99,1.03,0.94 \
  --explicit-group-sizes 30,30,30,30,30,30 \
  --explicit-cap-angle 0.25 \
  --explicit-theorem-cap-angle 0.28 \
  --explicit-init-mode paired \
  --explicit-theory-check strict \
  --explicit-auto-margin-ratio 0.05 \
  --gif-frames 60 \
  --fps 15 \
  --snapshot-count 5 \
  --snapshot-columns 5 \
  --dpi 135 \
  --output-dir "$SCRIPT_DIR/results"
