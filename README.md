# DualSQL

EditSQL with dual learning mechanism.

## Data Preparation

Download sparc dataset and glove.

* glove/glove.840B.300d.txt
* data/sparc

Download `nltk.punkt`.

## Preprocess

Remove unwanted attributes from json, and cast column names to the form of `table_name.column_name`.

```bash
python3 preprocess.py --dataset=sparc --remove_from
```

## Train

Train DualSQL and language models for utterance/query.

```bash
bash train.sh
```

## Evaluate

Calculate the question match and the interaction match.

```bash
# etype=match
python3 postprocess_eval.py --dataset=sparc --split=dev --pred_file results/pred.json --remove_from
```

## Experiment Results

