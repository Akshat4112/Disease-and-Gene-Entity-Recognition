from Preprocess import Preprocess
from NaiveBayes import NaiveBayes
from Evaluation import precisionrecall
from nltk import tokenize

train = '../data/ner-disease/train.iob'
test = '../data/ner-disease/test.iob' 
dev = '../data/ner-disease/dev.iob'
dev_predicted = '../data/ner-disease/dev-predicted.iob'

preprocess = Preprocess()
preprocess.text_to_data(filepath=train)
X, y = preprocess.preprocess_data()
preprocess.text_to_data(filepath=test)
Xtest, y_true = preprocess.preprocess_data()

nb = NaiveBayes()
nb.MultinomialNBTrain(X,y)

y_pred = []
for item in Xtest:
    pred = nb.MultinomialNBTest(item)
    y_pred.append(pred)

precisionrecall(y_pred, y_true)

# We have y_true
# We have y_pred
print(len(y_pred))
print(len(y_true))
print(y_pred[:10])
print(y_true[:10])
print(Xtest[:50])

y_pred = ['|B-DISEASE\n', '|O\n', '|B-DISEASE\n', '|B-DISEASE\n', '|B-DISEASE\n', '|O\n', '|B-DISEASE\n', '|B-DISEASE\n', '|O\n', '|I-DISEASE\n', '|O\n', '|B-DISEASE\n', '|I-DISEASE\n', '|O\n']
y_true = ['|O\n', '|O\n', '|O\n', '|O\n', '|O\n', '|O\n', '|O\n', '|B-DISEASE\n', '|I-DISEASE\n', '|I-DISEASE\n', '|O\n', '|B-DISEASE\n',  '|I-DISEASE\n', '|O\n']
TP = 0 
TN = 0
FP = 0
FN = 0
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
        print("True Entity: ", entity_true)
        print("Pred Entity: ", entity_predicted)
        print("Label True: ", Label_True)
        print("Label Predicted: ", Label_Predicted)
    entity_true = ''
    entity_predicted = ''
    Label_True = ''
    Label_Predicted = ''
print(TP)



















# TP = 0 
# FP = 0
# for i in range(len(y_true)):
#     if y_true[i] == y_pred[i] == '|B-DISEASE\n' and y_true[i+1] !='|I-DISEASE':
#         TP+=1
#     elif y_true[i] == y_pred[i] == '|B-DISEASE\n' and y_true[i+1] =='|I-DISEASE'==y_pred[i+1]:
#         TP+=1
#     elif y_true[i] != y_pred[i] :
#         FP+=1
# print(TP)
# print(FP)