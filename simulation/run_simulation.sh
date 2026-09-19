set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:-explicit}"

case "$MODEL" in
  standard|explicit|implicit) ;;
  *)
    echo "Usage: $0 [standard|explicit|implicit]" >&2
    exit 2
    ;;
esac

python "$SCRIPT_DIR/main.py" \
  --model "$MODEL" \
  --n 180 \
  --dim 3 \
  --beta 1.0 \
  --T 10 \
  --dt 0.05 \
  --seed 42 \
  --device auto \
  --standard-init l2 \
  --leaders 6 \
  --leader-radii 1.85,1.85,1.85,1.85,1.85,1.85 \
  --leader-weights 1.01,0.95,1.02,0.99,1.03,0.94 \
  --components 6 \
  --implicit-component-sizes 30,30,30,30,30,30 \
  --interaction-radius 1.5 \
  --coordinate-spacing 1.0 \
  --component-gap 6.0 \
  --implicit-cap-angle 0.75 \
  --gif-frames 60 \
  --fps 15 \
  --snapshot-count 5 \
  --snapshot-columns 5 \
  --dpi 135 \
  --output-dir "$SCRIPT_DIR/results"
