''' 
Project: Explaining LSTM-CRF models based NER Systems
Version: 0.1
Author: Akshat Gupta
'''

import pandas as pd

'''
To understand data properties
'''
class EDA():
    def __init__(self) -> None:
        pass
    
    #To read CSV
    def read_csv(self):
        self.df = pd.read_csv('../data/dfnew.csv')

    #To get facts about the dataset
    def EDA(self):
        print("Total number of sentences in the dataset: {:,}".format(self.df["Sentence"].nunique()))
        print("Total words in the dataset: {:,}".format(self.df.shape[0]))

        self.df["POS"].value_counts().plot(kind="bar", figsize=(10,5))
        self.df[self.df["Tag"]!="O"]["Tag"].value_counts().plot(kind="bar", figsize=(10,5))

        word_counts = self.df.groupby("Sentence")["Word"].agg(["count"])
        word_counts = word_counts.rename(columns={"count": "Word count"})
        word_counts.hist(bins=50, figsize=(8,6))
        
        MAX_SENTENCE = word_counts.max()[0]
        print("Longest sentence in the corpus contains {} words.".format(MAX_SENTENCE))

        longest_sentence_id = word_counts[word_counts["Word count"]==MAX_SENTENCE].index[0]
        print("ID of the longest sentence is {}.".format(longest_sentence_id))

        longest_sentence = self.df[self.df["Sentence"]==longest_sentence_id]["Word"].str.cat(sep=' ')
        print("The longest sentence in the corpus is:",longest_sentence)
        print("EDA Completed")
        
