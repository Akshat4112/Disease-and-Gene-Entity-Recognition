import numpy as np

from collections import Counter
from typing import List, Any

class Evaluation(object):
    def __init__(self) -> None:
        pass
def precisionrecall(y_pred: List[Any], y_true: List[Any]):
    # y_pred = []
    # y_true = []

    w, h = 3, 3
    confusionmatrix = [[0 for i in range(w)] for j in range(h)]  
    
    for i in range(len(y_pred)):  # i is the index in the data
        classpred = y_pred[i]
        classtruth = y_true[i]

        pred_class_to_index = {"|B-DISEASE\n": 0, "|I-DISEASE\n": 1, "|O\n": 2}
        true_class_to_index = {"|B-DISEASE\n": 0, "|I-DISEASE\n": 1, "|O\n": 2}

        row_index = true_class_to_index[classtruth]
        column_index = pred_class_to_index[classpred]

        confusionmatrix[row_index][column_index] += 1

    # finding precision and recall for each class
    # precision = (TP) / (TP + FP)
    # recall  = (TP) / (TP + FN)
    
    recall_c0 = confusionmatrix[0][0] / sum(confusionmatrix[0])  # we take the TP (column ind 0 and row ind 0) and divide by all row
    recall_c1 = confusionmatrix[1][1] / sum(confusionmatrix[1])  # we take the TP (column ind 1 and row ind 1) and divide by all row
    recall_c2 = confusionmatrix[2][2] / sum(confusionmatrix[2])  # we take the TP (column ind 2 and row ind 2) and divide by all row

    tot_recall_c = (recall_c0 + recall_c1 + recall_c2) / 3

    precision_c0 = confusionmatrix[0][0] / sum([confusionmatrix[i][0] for i in range(3)])
    precision_c1 = confusionmatrix[1][1] / sum([confusionmatrix[i][1] for i in range(3)])
    precision_c2 = confusionmatrix[2][2] / sum([confusionmatrix[i][2] for i in range(3)])
    
    # maybe we can generalize by another list comprehension for the numerator

    tot_precision_c = (precision_c0 + precision_c1 + precision_c2) / 3

    f1score_c0 = (2 * precision_c0 * recall_c0) / (precision_c0 + recall_c0)
    f1score_c1 = (2 * precision_c1 * recall_c1) / (precision_c1 + recall_c1)
    f1score_c2 = (2 * precision_c2 * recall_c2) / (precision_c2 + recall_c2)

    tot_f1score_c = (f1score_c0 + f1score_c1 + f1score_c2) / 3

    print("Precision and Recall...")
    print("                          Precision                               Recall")
    print("|B-DISEASE        "+ str(precision_c0)+"                   "+str(recall_c0))
    print("|I-DISEASE        "+ str(precision_c1)+"                   "+str(recall_c1))
    print("|O                "+ str(precision_c2)+"                   "+str(recall_c2))

    return "Evaluation Completed..."






