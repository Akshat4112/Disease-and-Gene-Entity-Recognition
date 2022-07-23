from eli5.lime import TextExplainer
from eli5.lime.samplers import MaskingTextSampler

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

index = 2000
label = labels[index]
text = sentences[index]
print(text)
print()
print(" ".join([f"{t} ({l})" for t, l in zip(text.split(), label)]))


for i, w in enumerate(text.split()):
    print(f"{i}: {w}")

explainer_generator = NERExplainerGenerator(model, word2idx, tag2idx, max_len)
word_index = 2
predict_func = explainer_generator.get_predict_function(word_index=word_index)
sampler = MaskingTextSampler(
    replacement="UNK",
    max_replace=0.7,
    token_pattern=None,
    bow=False
)
samples, similarity = sampler.sample_near(text, n_samples=4)
print(samples)
te = TextExplainer(
    sampler=sampler,
    position_dependent=True,
    random_state=42
)

clf = te.fit(text, predict_func)

te.explain_prediction(
    target_names=list(explainer_generator.idx2tag.values()),
    top_targets=3
)

def get_explaination(sentID):
  explainer = []
  index = sentID
  label = labels[index]
  text = sentences[index]
  print(text, label)
  explainer_generator = NERExplainerGenerator(model, word2idx, tag2idx, max_len)
  for wi in range(len(str(text))):
    word_index = wi
    predict_func = explainer_generator.get_predict_function(word_index=word_index)

    sampler = MaskingTextSampler(replacement="UNK",max_replace=0.7,token_pattern=None,bow=False)

    samples, similarity = sampler.sample_near(text, n_samples=4)
    print(samples)

    te = TextExplainer(sampler=sampler,position_dependent=True,random_state=42)

    clf = te.fit(text, predict_func)

    explaination = te.explain_prediction(target_names=list(explainer_generator.idx2tag.values()),top_targets=3)
    explainer.append(explaination)
  return explainer
sentID = 2000
explain = get_explaination(sentID)