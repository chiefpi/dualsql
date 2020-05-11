CUDA_VISIBLE_DEVICES=0 python3 scripts/run_lm.py \
    --primal \
    --train \
    --evaluate \
    --remove_from \
    --evaluate_split="dev" \
    --task_name="lm-p"
# CUDA_VISIBLE_DEVICES=0 python3 scripts/run_qg.py
# CUDA_VISIBLE_DEVICES=0 python3 scripts/run_sp.py
# CUDA_VISIBLE_DEVICES=0 python3 scripts/run_dual.py