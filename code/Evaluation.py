# Importing important libraries
import numpy as np
from collections import Counter
from typing import List, Any

# Evaluation class which takes prediction and true as a list and return Precision, Recall, and F1 score, token-based as well as entity-based.
class Evaluation(object):
    def __init__(self) -> None:
        pass

# Token-based Precision, Recall Implementation

def PrecisionRecall(y_pred: List[Any], y_true: List[Any]):
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

    #Printing Precision, Recall, and F1 Score. 
    tot_f1score_c = (f1score_c0 + f1score_c1 + f1score_c2) / 3
    print("Precision: ", tot_precision_c)
    print("Recall: ", tot_recall_c)
    print("F1 score: ", tot_f1score_c)
    
    return "Evaluation Completed..."

# Entity-Level Precision, Recall, and F1 score, it takes 2 list, ypred and y_true
def PrecisionRecallEntityLevel(Xtest: List[Any], y_pred: List[Any], y_true: List[Any]):

    # Initalizing metrics to 0
    TP = 0 
    TN = 0
    FP = 0
    FN = 0

    # We find the first encounter of B in true and pred, then we look at what follows it:
        # if it is I then we look for consecutive I's to match the entity with sequence
        # else if it is only B and a match we say it is a TP
        # if not then it is a false positive 
    # Otherwise it is a false negative
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i] == '|B-DISEASE\n':
            j = i+1
            z = i+1
            k = 0
            l = 0
            entity_true = ''.join(Xtest[i]) 
            entity_predicted = ''.join(Xtest[i]) 
            Label_True = ''.join(y_true[i]) 
            Label_Predicted = ''.join(y_pred[i]) 
            while y_true[j] == '|I-DISEASE\n':
                k+=1
                j+=1
                entity_true = entity_true + " " +  Xtest[j]
                Label_True = Label_True+ ''+ y_true[j]
            
            while y_pred[z] == '|I-DISEASE\n':
                l+=1
                z+=1
                entity_predicted = entity_predicted + " " +  Xtest[z]
                Label_Predicted = Label_Predicted+ ''+ y_pred[z]

            if (k==l):
                TP+=1
            else:
                FN+=1
            LTsp = Label_True.split()
            LPsp = Label_Predicted.split()
            if Label_True !=Label_Predicted:
                FP+=1
                    
        elif y_true[i]!=y_pred[i] !='|B-DISEASE\n':
            FN+=1
        
        entity_true = ''
        entity_predicted = ''
        Label_True = ''
        Label_Predicted = ''

    # Computing Precision, Recall, and F1 Score            
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1 = (2 * Precision * Recall) / (Precision + Recall)

    #Printing Precision, Recall, and F1 Score. 
    print("Precision: ", Precision)
    print("Recall: ", Recall)
    print("F1-Score: ", F1)


def PrecisionRecallEntityLevelGene(Xtest: List[Any], y_pred: List[Any], y_true: List[Any]):

    # Initalizing metrics to 0
    TP = 0 
    TN = 0
    FP = 0
    FN = 0

    # We find the first encounter of B in true and pred, then we look at what follows it:
        # if it is I then we look for consecutive I's to match the entity with sequence
        # else if it is only B and a match we say it is a TP
        # if not then it is a false positive 
    # Otherwise it is a false negative
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i] == '|B-PROTEIN\n':
            j = i+1
            z = i+1
            k = 0
            l = 0
            entity_true = ''.join(Xtest[i]) 
            entity_predicted = ''.join(Xtest[i]) 
            Label_True = ''.join(y_true[i]) 
            Label_Predicted = ''.join(y_pred[i]) 
            while y_true[j] == '|I-PROTEIN\n':
                k+=1
                j+=1
                entity_true = entity_true + " " +  Xtest[j]
                Label_True = Label_True+ ''+ y_true[j]
            
            while y_pred[z] == '|I-PROTEIN\n':
                l+=1
                z+=1
                entity_predicted = entity_predicted + " " +  Xtest[z]
                Label_Predicted = Label_Predicted+ ''+ y_pred[z]

            if (k==l):
                TP+=1
            else:
                FN+=1
            LTsp = Label_True.split()
            LPsp = Label_Predicted.split()
            if Label_True !=Label_Predicted:
                FP+=1
                    
        elif y_true[i]!=y_pred[i] !='|B-PROTEIN\n':
            FN+=1
        
        entity_true = ''
        entity_predicted = ''
        Label_True = ''
        Label_Predicted = ''

    # Computing Precision, Recall, and F1 Score            
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1 = (2 * Precision * Recall) / (Precision + Recall)

    #Printing Precision, Recall, and F1 Score. 
    print("Precision: ", Precision)
    print("Recall: ", Recall)
    print("F1-Score: ", F1)




