# DualSQL

EditSQL with dual learning mechanism.

## Data Preparation

Download the followings

* glove/glove.840B.300d.txt
* data/spider
* data/sparc
* data/cosql

Download `nltk.punkt`.

## Preprocess

Remove unwanted attributes from json, and modify column names in the format of `table_name.column_name`.

```bash
python preprocess.py --dataset=sparc --remove_from
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
python3 postprocess_eval.py --dataset=sparc --split=dev --pred_file log/pred.json --remove_from
```

## Experiment Results

