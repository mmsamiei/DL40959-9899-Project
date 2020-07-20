# DL40959-9899-Project

Required material for Deep Learning Course Project

Instructor: Mahdieh Soleymani

Course Page: [http://ce.sharif.edu/courses/98-99/2/ce719-1/](http://ce.sharif.edu/courses/98-99/2/ce719-1/)

## Prerequsites

```
pip3 install -r requirements.txt
```

## Usage

#### For Evaluating BLEU Score
```bash
python3 Evaluation/bleu_score.py --target-dataframe dataset.valid.csv --predicted-captions sentences.txt --ngram 1
```

#### For Evaluating Precision and IoU Scores

```bash
python3 Evaluation/comprehension_score.py --target-dataframe dataset.valid.csv --predicted-boxes boxes.txt
```
