import sys, argparse
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import numpy as np

def eval_bleu(target_df, predicated_sentences, ngram):
  list_of_references = []
  list_of_hypotheses = []
  for i in range(len(target_df)):
    image_fname = target_df.loc[i, 'image_fname']
    box = target_df.loc[i, 'box']
    similar_rows = target_df[ (target_df['image_fname'] == image_fname) & (target_df['box'] == box) ]
    refrences = [ sentence.split() for sentence in similar_rows['sentence'] ]
    hypotheses = predicated_sentences[i].split()
    list_of_references.append(refrences)
    list_of_hypotheses.append(hypotheses)
  weights = [1/ngram for _ in range(ngram)]
  return corpus_bleu(list_of_references, list_of_hypotheses, weights=weights)


def process_args(args):
    parser = argparse.ArgumentParser(description='Evaluate bleu score.')

    parser.add_argument('--target-dataframe', dest='target_file',
                        type=str, required=True,
                        help=(
                            'target dataframe file'
                        ))

    parser.add_argument('--predicted-captions', dest='predicted_file',
                        type=str, required=True,
                        help=(
                            'predicted captions file'
                        ))

    parser.add_argument('--ngram', dest='ngram',
                        type=int, required=True,
                        help=(
                            'ngram for blue'
                        ))

    parameters = parser.parse_args(args)
    return parameters


def main(args):
    parameters = process_args(args)

    target_file = parameters.target_file
    predicted_file = parameters.predicted_file
    ngram = parameters.ngram

    df = pd.read_csv(target_file)
    sentences = open(predicted_file).readlines()
    print("blue-{}: {}".format(ngram,eval_bleu(df, sentences, ngram)))

if __name__ == '__main__':
    main(sys.argv[1:])