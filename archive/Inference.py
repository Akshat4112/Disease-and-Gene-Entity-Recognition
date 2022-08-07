''' 
Project: Explaining LSTM-CRF models based NER Systems
Version: 0.1
Author: Akshat Gupta
'''
#Not used
# sentence = "Identification of APC2, a homologue of the adenomatous polyposis coli tumour suppressor."

# re_tok = re.compile(f"([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])")
# sentence = re_tok.sub(r"  ", sentence).split()

# padded_sentence = sentence + [word2index["--PADDING--"]] * (MAX_SENTENCE - len(sentence))
# padded_sentence = [word2index.get(w, 0) for w in padded_sentence]

# pred = ner_model.predict(np.array([padded_sentence]))
# pred = np.argmax(pred, axis=-1)

# retval = ""
# for w, p in zip(sentence, pred[0]):
#   retval = retval + "{:15}: {:5}".format(w, index2tag[p]) + "\n"
# print(retval)

