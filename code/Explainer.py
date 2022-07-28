''' 
Project: Explaining LSTM-CRF models based NER Systems
Version: 0.1
Author: Akshat Gupta
'''

import pickle
from eli5.lime import TextExplainer
from eli5.lime.samplers import MaskingTextSampler
from tensorflow.keras.utils import pad_sequences
from tensorflow import keras
import eli5
import pandas as pd
import ast

class NERExplainerGenerator(object):
    
    def __init__(self, model, word2idx, tag2idx, max_len):
        self.model = model
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.idx2tag = {v: k for k,v in tag2idx.items()}
        self.max_len = max_len
        
    def _preprocess(self, texts):
        X = [[self.word2idx.get(w, self.word2idx["UNK"]) for w in t.split()]
             for t in texts]
        X = pad_sequences(maxlen=self.max_len, sequences=X,
                          padding="post", value=self.word2idx["PAD"])
        return X
    
    def get_predict_function(self, word_index):
        def predict_func(texts):
            X = self._preprocess(texts)
            p = self.model.predict(X)
            return p[:,word_index,:]
        return predict_func


model  = keras.models.load_model('../models/ckpt1658660485.8331368.h5')
with open('../data/word2idx.pkl', "rb") as f:
    word2idx = pickle.load(f)

with open('../data/tag2idx.pkl', "rb") as f1:
    tag2idx = pickle.load(f1)

max_len = 114
explainer_generator = NERExplainerGenerator(model, word2idx, tag2idx, max_len)
word_index = 2
predict_func = explainer_generator.get_predict_function(word_index=word_index)
sampler = MaskingTextSampler(
    replacement="UNK",
    max_replace=0.7,
    token_pattern=None,
    bow=False
)

def explaination_generator(text):
    samples, similarity = sampler.sample_near(text, n_samples=4)
    print("Input is: ",samples)
    print("Fitting Explainer...")
    print("----------------------------------------------")
    te = TextExplainer(sampler=sampler,position_dependent=True,random_state=42)
    
    word_importance = {}
    word_importance["B"] = []
    word_importance["I"] = []
    word_importance["O"] = []

    try:
        clf = te.fit(text, predict_func)
        explaination = te.explain_prediction(target_names=list(explainer_generator.idx2tag.values()),top_targets=3)
        exp = eli5.formatters.format_as_dict(explaination)
        features = exp['targets']
        print("----------------------------------------------")
        print("Model Fitting Completed")
  
        for item in features:
            for ite in item['feature_weights']['pos']:
                word = ite['feature'].split()
                del(word[0])
                word = ''.join(word)
                if item['target'] == '|B-DISEASE\n':
                    word_importance["B"].append(word)           
                elif item['target'] == '|I-DISEASE\n':
                    word_importance["I"].append(word)           
                elif item['target'] == '|O\n':
                    word_importance["O"].append(word)           
    except Exception as e:
        print(e)

        # for ite in item['feature_weights']['neg']:
        #     word = ite['feature'].split()
        #     del(word[0])
        #     word = ''.join(word)
        #     if item['target'] == '|B-DISEASE\n':
        #         word_importance["B"].append(word)           
        #     elif item['target'] == '|I-DISEASE\n':
        #         word_importance["I"].append(word)           
        #     elif item['target'] == '|O\n':
        #         word_importance["O"].append(word)           
    return word_importance

#Loading the data from dataframe
df = pd.read_csv('../data/ner-disease/DatasetTrain.csv')
print(df.head())
sentences = []
temp = []

for i in range(len(df['Word']) -1):
    if df['Sentence'][i] == df['Sentence'][i+1]:
        temp.append(df['Word'][i])
    elif df['Sentence'][i] != df['Sentence'][i+1]:
        sent_temp = ''.join(str(temp))
        
        sentences.append(sent_temp)
        temp = []
    
    
# print(sentences) 

df_sents = pd.DataFrame(columns=['sentences'])
df_sents['sentences'] =  sentences
df_sents.to_csv('../data/ner-disease/sentences.csv')

words_final = []
for item in df_sents['sentences']:
    refined_sentence = ast.literal_eval(item)
    final_sentece = ' '.join(refined_sentence)

    # text = 'PCR amplification from genomic DNA and automated sequencing of the entire coding region ( 66 exons ) and splice junctions detected 77 mutations ( 85 % ) in 90 A-T chromosomes .'        
    wi = explaination_generator(final_sentece)
    print(wi)
    words_final.append(wi)

words_df = pd.DataFrame(columns=['dicts'])
words_df['dicts'] = words_final
words_df.to_csv('../data/ner-disease/words_dicts.csv')