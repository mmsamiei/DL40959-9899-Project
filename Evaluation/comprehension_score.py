import sys, argparse
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import numpy as np

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def eval_precision(target_df, predicated_boxes):
  '''
  target_df: dataframe,
  predicated_boxes: list of boxes
  '''
  l = []
  for idx in range(len(target_df)):
    correct_box = [float(s) for s in target_df.loc[idx,'box'][1:-1].split()]
    pred_box = predicated_boxes[idx]
    x_0_box, y_0_box = correct_box[0], correct_box[1]
    x_1_box = x_0_box + correct_box[2]
    y_1_box = y_0_box + correct_box[3] 
    
    x_0_pred, y_0_pred = pred_box[0], pred_box[1]
    x_1_pred = x_0_pred + pred_box[2]
    y_1_pred = y_0_pred + pred_box[3]
    correct_box = [x_0_box, y_0_box, x_1_box, y_1_box]
    pred_box = [x_0_pred, y_0_pred, x_1_pred, y_1_box]
    IoU = bb_intersection_over_union(correct_box, pred_box)
    l.append(IoU)
  correct_prediction = [int(x>=0.5) for x in l]
  avg_IoU = sum(l) / len(l)
  p_1 = sum(correct_prediction) / len(correct_prediction)
  return p_1, avg_IoU


def process_args(args):
    parser = argparse.ArgumentParser(description='Evaluate bleu score.')

    parser.add_argument('--target-dataframe', dest='target_file',
                        type=str, required=True,
                        help=(
                            'target dataframe file'
                        ))

    parser.add_argument('--predicted-boxes', dest='predicted_file',
                        type=str, required=True,
                        help=(
                            'predicted boxes file'
                        ))

    parameters = parser.parse_args(args)
    return parameters


def main(args):
    parameters = process_args(args)

    target_file = parameters.target_file
    predicted_file = parameters.predicted_file

    df = pd.read_csv(target_file)
    boxes = [line.split(',') for line in open(predicted_file).readlines()]
    boxes = [ [float(mm) for mm in instance]
        for instance in boxes
    ]
    p_1, avg_IoU = eval_precision(df, boxes)
    print("Precision: ", p_1)
    print("IoU: ", avg_IoU)

if __name__ == '__main__':
    main(sys.argv[1:])

