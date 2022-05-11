import numpy as np

class Evaluation(object):
    def __init__(self) -> None:
        pass
    def precision(y_pred, y_true):
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(len(y_pred)):
            if   (y_pred[i] == 0) & (y_true[i] == 0):
                TP += 1
            elif (y_pred[i] == 0) & (y_true[i] == 1):
                FP += 1
            elif (y_pred[i] == 1) & (y_true[i] == 1):
                TN += 1.
            else:
                FN += 1
        precision = (TP) / (TP + FP) 
        return precision


    def recall(y_pred, y_true):
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(len(y_pred)):
            if   (y_pred[i] == 0) & (y_true[i] == 0):
                TP += 1
            elif (y_pred[i] == 0) & (y_true[i] == 1):
                FP += 1
            elif (y_pred[i] == 1) & (y_true[i] == 1):
                TN += 1
            else:
                FN += 1
        recall = (TP) / (TP + FN)         
        return recall

    def f1score(y_pred, y_true):
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(len(y_pred)):
            if   (y_pred[i] == 0) & (y_true[i] == 0):
                TP += 1
            elif (y_pred[i] == 0) & (y_true[i] == 1):
                FP += 1
            elif (y_pred[i] == 1) & (y_true[i] == 1):
                TN += 1
            else:
                FN += 1

        precision = (TP) / (TP + FP) 
        recall    = (TP) / (TP + FN) 
        f1_score  = (2 * precision * recall) / (precision + recall)
        
        return f1_score

    def plot_confusion_matrix(y_pred, y_true):
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(len(y_pred)):
            if   (y_pred[i] == 0) & (y_true[i] == 0):
                TP += 1
            elif (y_pred[i] == 0) & (y_true[i] == 1):
                FP += 1
            elif (y_pred[i] == 1) & (y_true[i] == 1):
                TN += 1
            else:
                FN += 1

        precision = (TP) / (TP + FP) 
        recall    = (TP) / (TP + FN) 
        f1_score  = (2 * precision * recall) / (precision + recall)
        #Add labels to the Matrix- To be done
        cm = np.array([[TN, FP],
            [FN, TP]])
        return cm