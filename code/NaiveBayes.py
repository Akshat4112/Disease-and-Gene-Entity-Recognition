from collections import Counter
import math


class NaiveBayes(object):
    def __init__(self):
        return None

    def MultinomialNBTrain(self, X, y):
        print("Training Started...")
        self.X = X
        self.y = y
        count_classes = Counter(y)
        count_B = count_classes["|B-DISEASE\n"]
        count_I = count_classes["|I-DISEASE\n"]
        count_O = count_classes["|O\n"]
        doc_count = count_B + count_I + count_O
        # print(doc_count)

        self.features = {}
        self.features["B"] = {}
        self.features["I"] = {}
        self.features["O"] = {}

        self.priori_B = math.log(count_B / doc_count)
        self.priori_I = math.log(count_I / doc_count)
        self.priori_O = math.log(count_O / doc_count)

        B = []
        I = []
        O = []

        for item, label in zip(X, y):
            if label == "|B-DISEASE\n":
                B.append(item)
            elif label == "I-DISEASE\n":
                I.append(item)
            else:
                O.append(item)

        B_dict = Counter(B)
        I_dict = Counter(I)
        O_dict = Counter(O)

        for word, count in B_dict.items():
            self.features["B"][word] = math.log(
                (int(count) + 1) / (count_B + doc_count)
            )
        for word, count in I_dict.items():
            self.features["I"][word] = math.log(
                (int(count) + 1) / (count_I + doc_count)
            )
        for word, count in O_dict.items():
            self.features["O"][word] = math.log(
                (int(count) + 1) / (count_O + doc_count)
            )
        print("Training Completed...")

    def MultinomialNBTest(self, X):
        self.X = X
        p_B, p_I, p_O = 0, 0, 0

        p_B += self.priori_B
        p_I += self.priori_I
        p_O += self.priori_O

        for item in self.X:
            if item in self.features["B"]:
                p_B += self.features["B"][item]

        for item in self.X:
            if item in self.features["I"]:
                p_I += self.features["I"][item]

        for item in self.X:
            if item in self.features["O"]:
                p_O += self.features["O"][item]

        #     print(p_B, p_I, p_O)
        results = [p_B, p_I, p_O]
        max_value = max(results)
        max_index = results.index(max_value)
        #     print(max_index)
        if max_index == 0:
            return "|I-DISEASE\n"
        elif max_index == 1:
            return "|B-DISEASE\n"
        else:
            return "|O\n"
