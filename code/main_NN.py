''' 
Project: Explaining LSTM-CRF models based NER Systems
Version: 0.1
Author: Akshat Gupta
'''

import pandas as pd
from DataPreperation import Preprocess
from EDA_on_Data import EDA
from TrainNew import NeuralNetwork

# train = '../data/ner-disease/train.iob'
# test = '../data/ner-disease/test.iob' 
# dev = '../data/ner-disease/dev.iob'
# dev_predicted = '../data/ner-disease/dev-predicted.iob'


# preprocess = Preprocess()
# preprocess.text_to_data(filepath=train)
# preprocess.preprocess_data()
# preprocess.create_dataframe()
# preprocess.dump_csv()

# eda = EDA()
# eda.read_csv()
# eda.EDA()


data = pd.read_csv("../data/dfnew.csv", encoding="latin1").fillna(method="ffill")
words = list(set(data["Word"].values))
n_words = len(words)

tags = list(set(data["Tag"].values))
n_tags = len(tags)

getData = NeuralNetwork(data)
sentences = getData.sentences
getData.Data_Encoding()
print("Data Encoding is Completed...")
getData.LSTM_NN()
print("Trainign NN is completed...")
getData.Training_Plots()
print("Curves Plotted and Stored in the Directory...")

