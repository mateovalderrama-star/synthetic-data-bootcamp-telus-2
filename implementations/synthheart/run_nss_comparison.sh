#!/usr/bin/env bash
# Runs every NeMo Safe Synthesizer arm, then scores and compares after each one, so results build up as it goes.
# Finished steps are skipped, so launching it again resumes where it stopped.
#
# Run on the AI Factory, from implementations/synthheart, after setup_nss.sh and the smoke run:
#     nohup bash run_nss_comparison.sh > logs/nss_comparison.log 2>&1 &
#     tail -f logs/nss_comparison.log
#
# ARMS="default dp8" bash run_nss_comparison.sh     # choose the arms (default: all three)
set -uo pipefail   # no -e: one failed arm must not stop the others
cd "$(dirname "$0")"
mkdir -p logs
NSS=.venv-nss/bin/python
TRAIN_ROWS=$(( $(wc -l < data/cardio/cardio.csv) - 1 ))

for arm in ${ARMS:-default dp8 3pass}; do
    case "$arm" in
        default) args=() ;;   # NSS as shipped: SmolLM3-3B + LoRA, 25,000 training records, no differential privacy
        dp8)     args=(--set privacy.dp_enabled=true --set privacy.epsilon=8.0   # DP-SGD, NVIDIA's default budget
                       --set training.batch_size=16 --set generation.use_structured_generation=true) ;;  # their DP tutorial settings
        3pass)   args=(--set training.num_input_records_to_sample=$(( 3 * TRAIN_ROWS ))) ;;  # ~3 passes over train instead of ~0.6
        *)       echo "unknown arm: $arm"; continue ;;
    esac
    echo "=== $(date '+%F %T') nss_$arm ==="
    "$NSS" 05_nss_generate.py --name "nss_$arm" ${args[@]+"${args[@]}"} 2>&1 | tee "logs/05_nss_$arm.log"
    echo "=== $(date '+%F %T') score + compare ==="
    uv run python 06_compare_generators.py
done
echo "=== $(date '+%F %T') done. Results: results/comparison/comparison.md ==="
