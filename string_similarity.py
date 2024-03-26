import numpy as np
import string

alphabet = list(string.printable)


class StringSimilarity:

    def __init__(self, source='', target=''):
        self.symbols = [] if len(alphabet) == 0 else alphabet
        self.source = source.lower()
        self.target = target.lower()

    def symbol_vector(self, value=''):
        output = np.zeros(len(self.symbols))
        output[self.symbols.index(value)] = 1
        return output

    def sentence_vector(self, value=''):
        output = []
        i = 1
        for s in value:
            if s in self.symbols:
                tmp = self.symbol_vector(value=s)
                output.append(tmp)
                i = i + 1
        return np.array(output)

    def run(self):
        v1 = self.sentence_vector(value=self.source)
        v2 = self.sentence_vector(value=self.target)
        v = np.dot(v1, v2.T)
        tmp = []
        a = int(np.sum(v.shape)/2)
        for k in range(-a, a, 1):
            tmp.append((np.sum(np.diag(v, k=k)) /
                        np.min(v.shape)))
        tmp = np.array(tmp)
        tmp[::-1].sort()
        result = max(tmp)
        for a in range(1, len(tmp)):
            result += tmp[a]/(a+1)
        # correction
        if result > 1.0:
            decimal_part = result % 1
            result = result - decimal_part*(1+decimal_part)
        # print('Result : ', result)
        return result
