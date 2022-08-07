''' 
Project: Explaining LSTM-CRF models based NER Systems
Version: 0.1
Author: Akshat Gupta
'''

# !wget https://setup.johnsnowlabs.com/nlu/colab.sh -O - | bash
import nlu
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

'''
To create BioBERT Embeddings
'''
class BioBERTEmbedding():
    def __init__(self) -> None:
        pass

    def Get_BioBertEmbedding():
        df = pd.read_csv('../data/ner-disease/DatasetTrain.csv')

        pipe = nlu.load('biobert pos') # emotion
        df['text'] = df['Word']
        df_b = df[df['Tag'] == '|B-DISEASE\n']
        df_i = df[df['Tag'] == '|I-DISEASE\n']
        df_o = df[df['Tag'] == '|O\n']

        df_b = df_b[:4000]
        df_i = df_i[:4000]
        df_o = df_o[:6000]

        df = pd.concat([df_b, df_o, df_i], ignore_index=True)
        predictions = pipe.predict(df[['text','Tag']].iloc[0:14000], output_level='token')

        # Make a matrix from the vectors in the np_array column via list comprehension
        mat = np.matrix([x for x in predictions.word_embedding_biobert])
        model = TSNE(n_components=2) #n_components means the lower dimension
        low_dim_data = model.fit_transform(mat)
        print('Lower dim data has shape',low_dim_data.shape)

        tsne_df =  pd.DataFrame(low_dim_data, predictions.Tag)
        tsne_df.columns = ['x','y']
        ax = sns.scatterplot(data=tsne_df, x='x', y='y', hue=tsne_df.index)
        ax.set_title('T-SNE BIOBERT Embeddings based on LIME and GALE, colored by Named Entity')
