from Preprocess import Preprocess
from NaiveBayes import NaiveBayes

# from Evaluation import precisionrecall
# from nltk import tokenize

train = "../data/ner-disease/train.iob"
test = "../data/ner-disease/test.iob"
dev = "../data/ner-disease/dev.iob"
dev_predicted = "../data/ner-disease/dev-predicted.iob"

preprocess = Preprocess()
preprocess.text_to_data(filepath=train)
X, y = preprocess.preprocess_data()
preprocess.text_to_data(filepath=test)
Xtest, y_true = preprocess.preprocess_data()

nb = NaiveBayes()
nb.MultinomialNBTrain(X, y)

y_pred = []
for item in Xtest:
    pred = nb.MultinomialNBTest(item)
    y_pred.append(pred)


"""
y_pred = [
    "|B-DISEASE\n",
    "|O\n",
    "|B-DISEASE\n",
    "|I-DISEASE\n",
    "|B-DISEASE\n",
    "|O\n",
    "|B-DISEASE\n",
    "|B-DISEASE\n",
    "|O\n",
    "|I-DISEASE\n",
    "|O\n",
    "|B-DISEASE\n",
    "|I-DISEASE\n",
    "|O\n",
]
y_true = [
    "|B-DISEASE\n",
    "|O\n",
    "|O\n",
    "|O\n",
    "|O\n",
    "|O\n",
    "|O\n",
    "|B-DISEASE\n",
    "|I-DISEASE\n",
    "|I-DISEASE\n",
    "|O\n",
    "|B-DISEASE\n",
    "|I-DISEASE\n",
    "|O\n",
]
"""

TP = 0
TN = 0
FP = 0
FN = 0


def createwordsequencesdisease(y):

    currentword = None
    dropi = False
    wordslist = []
    FP = 0

    for i in range(len(y)):
        if dropi:  # in case we encounter a beginning i first state
            if y[i] == "|I-DISEASE\n":
                continue
            elif y[i] == "|O\n":
                dropi = False
            elif y[i] == "|B-DISEASE\n":
                currentword = [i]
                dropi = False

        elif not currentword:  # not currently word, second state
            if y[i] == "|O\n":
                continue
            elif y[i] == "|B-DISEASE\n":
                currentword = [i]
            elif y[i] == "|I-DISEASE\n":
                dropi = True
                FP += 1

        else:  # yes we are processing a current word, third state
            if y[i] == "|O\n":
                wordslist.append(tuple(currentword))  # store partial result
                currentword = (
                    None  # we are done with the current word so we set it back to none
                )
            elif y[i] == "|B-DISEASE\n":
                wordslist.append(tuple(currentword))  # store partial result
                currentword = [
                    i
                ]  # & we stay in state because we are seeing another word
            else:  # if i is i-disease label
                currentword.append(
                    i
                )  # we store in the tuple of the current word as many i as we encounter

    return wordslist, FP


wordseqtrue, _ = createwordsequencesdisease(y_true)
wordseqpred, FP = createwordsequencesdisease(y_pred)
wordseqtrueset = set(wordseqtrue)
wordseqpredset = set(wordseqpred)

print(wordseqtrue)
print(wordseqpred)

intersection = wordseqtrueset.intersection(wordseqpredset)
list(intersection)
TP = len(intersection)
FP += len(wordseqpredset - intersection)
FN = len(wordseqtrueset - intersection)


print("TP: ", TP)
print("FP: ", FP)
print("FN: ", FN)

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = (2 * Precision * Recall) / (Precision + Recall)

print("Precision: ", Precision)
print("Recall: ", Recall)
print("F1-Score: ", F1)
