#Import our custom classes for Preprocess, Naive Bayes, and Evaluation

from Preprocess import Preprocess
from NaiveBayes import NaiveBayes
from Evaluation import PrecisionRecallEntityLevel, PrecisionRecall, PrecisionRecallEntityLevelGene

# Streamlit is to create the Web App
import streamlit as st

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

print("Results for Disease Level NER:")
PrecisionRecallEntityLevel(Xtest, y_pred, y_true)

# Training and Getting Entities for Gene Level Data

train = '../data/ner-gene/train.iob'
test = '../data/ner-gene/test.iob' 
dev = '../data/ner-gene/dev.iob'
dev_predicted = '../data/ner-gene/dev-predicted.iob'


preprocess = Preprocess()
preprocess.text_to_data(filepath=train)
X, y = preprocess.preprocess_data()
preprocess.text_to_data(filepath=test)
Xtest, y_true = preprocess.preprocess_data()

nb = NaiveBayes()
nb.MultinomialNBTrainGene(X,y)

y_pred = []
for item in Xtest:
    pred = nb.MultinomialNBTestGene(item)
    y_pred.append(pred)
print("Results for Gene Level NER:")
PrecisionRecallEntityLevelGene(Xtest, y_pred, y_true)

st.title("NER for Disease and Gene")
raw_text = st.text_area("Your Text","Enter Text Here")
tokens = raw_text.split()
custom_pred = []

if (st.button('Submit')):
    for item in tokens:
        pred = nb.MultinomialNBTest(item)
        custom_pred.append(pred)
    st.text(custom_pred)



