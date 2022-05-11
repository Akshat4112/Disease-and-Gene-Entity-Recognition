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
