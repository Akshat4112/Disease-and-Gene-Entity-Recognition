from typing_extensions import Self


class Preprocess():
    def __init__(self):
        self.data = []
        pass
        
    def text_to_data(self, filepath):
        self.filepath = filepath
        file = open(filepath, 'r')
        
        for line in file.readlines():
            self.data.append(line)
        
    
    def preprocess_data(self):
        X = []
        y = []

        for item in self.data:
            temp = item.split('\t')
            if len(item) > 1:
                X.append(temp[0])
                y.append(temp[1])
        return X, y 


train = '/content/train.iob'
test = '/content/test.iob' 
dev = '/content/dev.iob'
dev_predicted = '/content/dev-predicted.iob'


preprocess = Preprocess()
preprocess.text_to_data(filepath=train)
X, y = preprocess.preprocess_data()
preprocess.text_to_data(filepath=test)
Xtest, y_true = preprocess.preprocess_data()


df = pd.DataFrame(columns=['Sentence', 'Word', 'POS', 'Tag'])
df['Word'] = X
df['Tag'] = y

for i in range(len(df['Word'])):
    item = df['Word'][i]
    tag = nltk.pos_tag([item])
    df['POS'][i] = tag[0][1]

df['Sentence'][0] = 'Sentence: '+ str(1)
k = 2

for i in range(len(df['Word'])):
    if df['Word'][i] == '.':
        df['Sentence'][i+1] = 'Sentence: ' + str(k)
        k+=1        

 dfnew = df.copy()

        dfnew = dfnew.fillna(method="ffill")
dfnew["Sentence"] = dfnew["Sentence"].apply(lambda s: s[9:])
# dfnew["Sentence"] = dfnew["Sentence"].astype("int32")
dfnew.head()

# dfnew.to_csv('../input/ner-disease/DatasetTrain.csv')