# DualSQL

EditSQL with dual learning mechanism.

## Data Preparation

Download the following datasets (excluding database)

* glove/glove.840B.300d.txt
* data/spider
* data/sparc
* data/cosql

Download `nltk.punkt`.

## Preprocess

```bash
python preprocess.py --dataset=cosql --remove_from
```

## Train

```bash
bash train.sh
```

## Evaluate

Calculate the question match and the interaction match.

```bash
# etype=match
python3 postprocess_eval.py --dataset=cosql --split=dev --pred_file log/valid_use_predicted_queries_predictions.json --remove_from
```

## Experiment Results

