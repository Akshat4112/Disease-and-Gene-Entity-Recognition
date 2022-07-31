import nlu
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns

pipe = nlu.load('biobert')

#Read the csv
df = pd.read_csv('../data/ner-disease/DatasetTrain.csv')
df['text'] = df['Word']
# NLU to gives us one row per embedded word by specifying the output level
predictions = pipe.predict(df[['text','Tag']], output_level='token')
print(predictions)
predictions.dropna(how='any', inplace=True)

mat = np.matrix([x for x in predictions.biobert_embeddings])

model = TSNE(n_components=2)
low_dim_data = model.fit_transform(mat)
print('Lower dim data has shape',low_dim_data.shape)

tsne_df =  pd.DataFrame(low_dim_data, predictions.pos)
ax = sns.scatterplot(data=tsne_df, x=0, y=1, hue=tsne_df.index)
ax.set_title('T-SNE BIOBERT Embeddings, colored by Part of Speech Tag')