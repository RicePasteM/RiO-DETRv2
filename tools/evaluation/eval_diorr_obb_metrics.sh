#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${PYTHON:-python}
GPU=${GPU:-4}
BATCH_SIZE=${BATCH_SIZE:-4}

cd "$REPO_DIR"

LOG_DIR=logs/diorr_obb_metrics
OUTPUT_ROOT=outputs/diorr_obb_metrics
SUMMARY_FILE="$OUTPUT_ROOT/summary.tsv"
IOU_THRS='[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]'

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"
printf 'model\tcheckpoint\tAP50\tAP75\tAP50_95\n' > "$SUMMARY_FILE"

models=(n l m s x)
checkpoints=(
  outputs/riov2_hgnetv2_n_diorr/best_stg2.pth
  outputs/riov2_hgnetv2_l_diorr/best_stg2.pth
  outputs/riov2_hgnetv2_m_diorr/best_stg2.pth
  outputs/riov2_hgnetv2_s_diorr/best_stg2.pth
  outputs/riov2_hgnetv2_x_diorr/best_stg1.pth
)

for i in "${!models[@]}"; do
  model=${models[$i]}
  checkpoint=${checkpoints[$i]}
  config="configs/rio-diorr/riov2_hgnetv2_${model}_diorr.yml"
  output_dir="$OUTPUT_ROOT/riov2_hgnetv2_${model}_diorr"
  log_file="$LOG_DIR/riov2_hgnetv2_${model}_diorr.log"

  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $REPO_DIR/$checkpoint" >&2
    exit 1
  fi

  echo "Evaluating RiOv2-${model^^} on GPU $GPU"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 nice -n 10 "$PYTHON" train.py \
    -c "$config" \
    --test-only \
    -r "$checkpoint" \
    --output-dir "$output_dir" \
    -u \
    "evaluator.iou_thrs=$IOU_THRS" \
    "val_dataloader.total_batch_size=$BATCH_SIZE" \
    > "$log_file" 2>&1

  metrics=$(grep '^OBB_METRICS ' "$log_file" | tail -n 1)
  if [[ -z "$metrics" ]]; then
    echo "Metrics line missing from $log_file" >&2
    exit 1
  fi
  ap50=$(sed -n 's/.*AP50=\([^ ]*\).*/\1/p' <<< "$metrics")
  ap75=$(sed -n 's/.*AP75=\([^ ]*\).*/\1/p' <<< "$metrics")
  ap50_95=$(sed -n 's/.*AP50_95=\([^ ]*\).*/\1/p' <<< "$metrics")
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${model^^}" "$checkpoint" "$ap50" "$ap75" "$ap50_95" >> "$SUMMARY_FILE"
  echo "$metrics"
done

echo "Summary: $REPO_DIR/$SUMMARY_FILE"
