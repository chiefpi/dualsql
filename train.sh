# utterance lm
python3 scripts/run_lm.py \
    --primal \
    --train \
    --evaluate \
    --remove_from \
    --tie_weights \
    --evaluate_split="valid" \
    --task_name="lm-u"

# query lm
python3 scripts/run_lm.py \
    --train \
    --evaluate \
    --remove_from \
    --tie_weights \
    --evaluate_split="valid" \
    --task_name="lm-q"

# supervise primal model
python3 scripts/run_sp.py \
    --primal \
    --train \
    --force \
    --freeze \
    --evaluate \
    --remove_from \
    --evaluate_split="valid" \
    --pred_file="results/sp.json" \
    --task_name="dual"

# supervise dual model
python3 scripts/run_sp.py \
    --train \
    --force \
    --freeze \
    --evaluate \
    --remove_from \
    --evaluate_split="valid" \
    --pred_file="results/qg.json" \
    --task_name="dual"

# dual learning
# python3 scripts/run_dual.py