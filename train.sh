CUDA_VISIBLE DEVICES=0 python3 train.py \
    --dataset="sparc" \
    --model="cdseq2seq"

python3 eval.py \
    --gold="" \
    --pred="" \
    --db="data/databse" \
    --table="" \
    --etype=""