from Preprocess import Preprocess
from NaiveBayes import NaiveBayes

train = '../data/ner-disease/train.iob'
test = '../data/ner-disease/test.iob' 
dev = '../data/ner-disease/dev.iob'
dev_predicted = '../data/ner-disease/dev-predicted.iob'

preprocess = Preprocess()
preprocess.text_to_data(filepath=train)
X, y = preprocess.preprocess_data()
preprocess.text_to_data(filepath=test)
Xtest, y_test = preprocess.preprocess_data()

nb = NaiveBayes()
nb.MultinomialNBTrain(X,y)