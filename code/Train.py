all_words = list(set(dfnew["Word"].values))
all_tags = list(set(dfnew["Tag"].values))

print("Number of unique words: {}".format(dfnew["Word"].nunique()))
print("Number of unique tags : {}".format(dfnew["Tag"].nunique()))

word2index = {word: idx + 2 for idx, word in enumerate(all_words)}

word2index["--UNKNOWN_WORD--"]=0

word2index["--PADDING--"]=1

index2word = {idx: word for word, idx in word2index.items()}

for k,v in sorted(word2index.items(), key=operator.itemgetter(1))[:10]:
    print(k,v)

test_word = "examinations"

test_word_idx = word2index[test_word]
test_word_lookup = index2word[test_word_idx]

print("The index of the word {} is {}.".format(test_word, test_word_idx))
print("The word with index {} is {}.".format(test_word_idx, test_word_lookup))

tag2index = {tag: idx + 1 for idx, tag in enumerate(all_tags)}
tag2index["--PADDING--"] = 0

index2tag = {idx: word for word, idx in tag2index.items()}

def to_tuples(data):
    iterator = zip(data["Word"].values.tolist(),
                   data["POS"].values.tolist(),
                   data["Tag"].values.tolist())
    return [(word, pos, tag) for word, pos, tag in iterator]

sentences = dfnew.groupby("Sentence").apply(to_tuples).tolist()

print(sentences[0])

X = [[word[0] for word in sentence] for sentence in sentences]
y = [[word[2] for word in sentence] for sentence in sentences]
print("X[0]:", X[0])
print("y[0]:", y[0])

X = [[word2index[word] for word in sentence] for sentence in X]
y = [[tag2index[tag] for tag in sentence] for sentence in y]
print("X[0]:", X[0])
print("y[0]:", y[0])



X = [sentence + [word2index["--PADDING--"]] * (MAX_SENTENCE - len(sentence)) for sentence in X]
y = [sentence + [tag2index["--PADDING--"]] * (MAX_SENTENCE - len(sentence)) for sentence in y]
print("X[0]:", X[0])
print("y[0]:", y[0])

TAG_COUNT = len(tag2index)
y = [ np.eye(TAG_COUNT)[sentence] for sentence in y]
print("X[0]:", X[0])
print("y[0]:", y[0])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)

print("Number of sentences in the training dataset: {}".format(len(X_train)))
print("Number of sentences in the test dataset : {}".format(len(X_test)))


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

WORD_COUNT = len(index2word)
DENSE_EMBEDDING = 50
LSTM_UNITS = 64
LSTM_DROPOUT = 0.1
DENSE_UNITS = 128
BATCH_SIZE = 256
MAX_EPOCHS = 10


input_layer = layers.Input(shape=(MAX_SENTENCE,))

model = layers.Embedding(WORD_COUNT, DENSE_EMBEDDING, embeddings_initializer="uniform", input_length=MAX_SENTENCE)(input_layer)

model = layers.Bidirectional(layers.LSTM(LSTM_UNITS, recurrent_dropout=LSTM_DROPOUT, return_sequences=True))(model)

model = layers.TimeDistributed(layers.Dense(DENSE_UNITS, activation="relu"))(model)

crf_layer = CRF(units=TAG_COUNT)
output_layer = crf_layer(model)
ner_model = Model(input_layer, output_layer)


loss = losses.crf_loss
acc_metric = metrics.crf_accuracy
opt = optimizers.Adam(lr=0.001)

ner_model.compile(optimizer=opt, loss=loss, metrics=[acc_metric])

ner_model.summary()

history = ner_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, validation_split=0.1, verbose=2)

plot_history(history.history)

y_pred = ner_model.predict(X_test)

y_pred = np.argmax(y_pred, axis=2)

y_test = np.argmax(y_test, axis=2)

accuracy = (y_pred == y_test).mean()

print("Accuracy: {:.4f}/".format(accuracy))

def tag_conf_matrix(cm, tagid):
  tag_name = index2tag[tagid]
  print("Tag name: {}".format(tag_name))
  print(cm[tagid])
  tn, fp, fn, tp = cm[tagid].ravel()
  tag_acc = (tp + tn) / (tn + fp + fn + tp)
  print("Tag accuracy: {:.3f} \n".format(tag_acc))

matrix = multilabel_confusion_matrix(y_test.flatten(), y_pred.flatten())

tag_conf_matrix(matrix, 0)
tag_conf_matrix(matrix, 1)
tag_conf_matrix(matrix, 2)
tag_conf_matrix(matrix, 3)

#Another Implementation for Train

from tqdm import tqdm, trange

# data = pd.read_csv("/content/ner_dataset.csv", encoding="latin1").fillna(method="ffill")
data = pd.read_csv("/content/DatasetTrain.csv", encoding="latin1").fillna(method="ffill")
data.tail(10)

words = list(set(data["Word"].values))
n_words = len(words); n_words

tags = list(set(data["Tag"].values))
n_tags = len(tags); n_tags


class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


getter = SentenceGetter(data)
sentences = getter.sentences


labels = [[s[2] for s in sent] for sent in sentences]
sentences = [" ".join([s[0] for s in sent]) for sent in sentences]
sentences[0]
print(labels[0])
from collections import Counter
from keras.preprocessing.sequence import pad_sequences

word_cnt = Counter(data["Word"].values)
vocabulary = set(w[0] for w in word_cnt.most_common(5000))
max_len = 50
word2idx = {"PAD": 0, "UNK": 1}
word2idx.update({w: i for i, w in enumerate(words) if w in vocabulary})
tag2idx = {t: i for i, t in enumerate(tags)}


X = [[word2idx.get(w, word2idx["UNK"]) for w in s.split()] for s in sentences]


X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=word2idx["PAD"])


y = [[tag2idx[l_i] for l_i in l] for l in labels]

y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["|O\n"])

from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, shuffle=False)
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, SpatialDropout1D, Bidirectional
word_input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words, output_dim=50, input_length=max_len)(word_input)
model = SpatialDropout1D(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)
model = Model(word_input, out)
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
history = model.fit(X_tr, y_tr.reshape(*y_tr.shape, 1),
                    batch_size=32, epochs=5,
                    validation_split=0.1, verbose=1)
