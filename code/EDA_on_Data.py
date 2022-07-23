print("Total number of sentences in the dataset: {:,}".format(dfnew["Sentence"].nunique()))
print("Total words in the dataset: {:,}".format(dfnew.shape[0]))

dfnew["POS"].value_counts().plot(kind="bar", figsize=(10,5));

dfnew[dfnew["Tag"]!="O"]["Tag"].value_counts().plot(kind="bar", figsize=(10,5))

word_counts = dfnew.groupby("Sentence")["Word"].agg(["count"])
word_counts = word_counts.rename(columns={"count": "Word count"})
word_counts.hist(bins=50, figsize=(8,6));

MAX_SENTENCE = word_counts.max()[0]
print("Longest sentence in the corpus contains {} words.".format(MAX_SENTENCE))

longest_sentence_id = word_counts[word_counts["Word count"]==MAX_SENTENCE].index[0]
print("ID of the longest sentence is {}.".format(longest_sentence_id))

longest_sentence = dfnew[dfnew["Sentence"]==longest_sentence_id]["Word"].str.cat(sep=' ')
print("The longest sentence in the corpus is:\n")
print(longest_sentence)
