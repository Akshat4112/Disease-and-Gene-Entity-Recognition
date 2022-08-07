''' 
Project: Explaining LSTM-CRF models based NER Systems
Version: 0.1
Author: Akshat Gupta
'''


from typing_extensions import Self
import pandas as pd
import nltk

'''
To Preprocess the data to create x and y mappings
'''
class Preprocess():
    def __init__(self):
        self.data = []
        pass
    #To read line by line data and create data list
    def text_to_data(self, filepath):
        self.filepath = filepath
        file = open(filepath, 'r')
        
        for line in file.readlines():
            self.data.append(line)
        
    #To create x and y mappings of data
    def preprocess_data(self):
        self.X = []
        self.y = []

        for item in self.data:
            temp = item.split('\t')
            if len(item) > 1:
                self.X.append(temp[0])
                self.y.append(temp[1])
        print("Preprocess Completed...")
        print("Now Creating Dataframe...")
        
    #To create dataframe with POS tag for words
    def create_dataframe(self):
        self.df = pd.DataFrame(columns=['Sentence', 'Word', 'POS', 'Tag'])
        self.df['Word'] = self.X
        self.df['Tag'] = self.y

        for i in range(len(self.df['Word'])):
            item = self.df['Word'][i]
            tag = nltk.pos_tag([item])
            self.df['POS'][i] = tag[0][1]

        self.df['Sentence'][0] = 'Sentence: '+ str(1)
        k = 2

        for i in range(len(self.df['Word'])):
            if self.df['Word'][i] == '.':
                self.df['Sentence'][i+1] = 'Sentence: ' + str(k)
                k+=1        

        self.dfnew = self.df.copy()
        self.dfnew = self.dfnew.fillna(method="ffill")
        self.dfnew["Sentence"] = self.dfnew["Sentence"].apply(lambda s: s[9:])
        print(self.dfnew.head())
        print("DataFrame Created...")

    #To store csv for the dataframe in data
    def dump_csv(self):
        self.dfnew.to_csv('../data/dfnew.csv')
        print("CSV Dumped in data Directory")