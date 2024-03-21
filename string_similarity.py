import numpy as np
import string

alphabet = list(string.printable)


class StringSimilarity:

    def __init__(self, source='', target=''):
        # print('Bill Similarity')
        self.symbols = [] if len(alphabet) == 0 else alphabet
        self.source = source.lower()
        self.target = target.lower()

    def symbol_vector(self, value=''):
        output = np.zeros(len(self.symbols))
        output[self.symbols.index(value)] = 1
        return output

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def sentence_vector(self, value=''):
        output = []
        i = 1
        for s in value:
            if s in self.symbols:
                tmp = self.symbol_vector(value=s)
                output.append(tmp)
                i = i + 1
        return np.array(output)

    def sigmoid(self, value=0.0):
        return 1 - (1 / (1 + np.tanh(value)))

    def run(self):
        v1 = self.sentence_vector(value=self.source)
        v2 = self.sentence_vector(value=self.target)
        v = np.dot(v1, v2.T)
        tmp = []
        for k in range(-v.shape[0]+1, v.shape[0], 1):
            tmp.append((np.sum(np.diag(v, k=k)) /
                        np.mean(v.shape)))
        tmp = np.array(tmp)
        max_pos = np.argmax(tmp)
        recall = np.max(tmp[max_pos:])
        return 2 * recall / (1 + recall)
