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

Remove unnecessary attributes from json.

```bash
python preprocess.py --dataset=sparc --remove_from
```

## Train

```bash
bash train.sh
```

## Evaluate

Calculate the question match and the interaction match.

```bash
# etype=match
python3 postprocess_eval.py --dataset=sparc --split=dev --pred_file log/todo.json --remove_from
```

TODO: pred_file name

## Experiment Results

