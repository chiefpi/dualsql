# primal lm
python3 scripts/run_lm.py \
    --primal \
    --train \
    --evaluate \
    --remove_from \
    --tie_weights \
    --evaluate_split="valid" \
    --task_name="lm-p"

# dual lm
python3 scripts/run_lm.py \
    --train \
    --evaluate \
    --remove_from \
    --tie_weights \
    --evaluate_split="valid" \
    --task_name="lm-d"

# supervise primal model
python3 scripts/run_sp.py \
    --primal \
    --train \
    --force \
    --freeze \
    --evaluate \
    --remove_from \
    --evaluate_split="valid" \
    --task_name="dual"

# supervise dual model
python3 scripts/run_sp.py \
    --train \
    --force \
    --freeze \
    --evaluate \
    --remove_from \
    --evaluate_split="valid" \
    --task_name="dual"

# dual learning
# python3 scripts/run_dual.py
