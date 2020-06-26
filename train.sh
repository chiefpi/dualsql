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

# supervise primal model
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_sp.py \
    --primal \
    --train \
    --force \
    --freeze \
    --evaluate \
    --remove_from \
    --evaluate_split="valid" \
    --task_name="sp"

# supervise dual model
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_sp.py \
    --train \
    --force \
    --freeze \
    --evaluate \
    --remove_from \
    --evaluate_split="valid" \
    --task_name="sp"

# dual learning
# CUDA_VISIBLE_DEVICES=0 python3 scripts/run_dl.py
