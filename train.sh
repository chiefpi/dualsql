# primal lm
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_lm.py \
    --primal \
    --train \
    --evaluate \
    --remove_from \
    --tie_weights \
    --evaluate_split="valid" \
    --task_name="lm-p"

# dual lm
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_lm.py \
    --train \
    --evaluate \
    --remove_from \
    --tie_weights \
    --evaluate_split="valid" \
    --task_name="lm-d"

# CUDA_VISIBLE_DEVICES=0 python3 scripts/run_qg.py
# CUDA_VISIBLE_DEVICES=0 python3 scripts/run_sp.py
# CUDA_VISIBLE_DEVICES=0 python3 scripts/run_dual.py
