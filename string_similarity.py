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
        trace = v.diagonal().sum()
        links = []
        for i in range(len(v)):
            r_pointer = i
            j = 1
            while (j < len(v[i])):
                status = v[r_pointer][j-1]
                if status == v[r_pointer][j]:
                    links.append((status, v[r_pointer][j]))
                if r_pointer + 1 < len(v):
                    if status == v[r_pointer+1][j]:
                        links.append((status, v[r_pointer+1][j]))
                    if v[r_pointer][j] == v[r_pointer+1][j]:
                        links.append((v[r_pointer][j], v[r_pointer+1][j]))
                    if v[r_pointer+1][j-1] == v[r_pointer][j]:
                        links.append((v[r_pointer+1][j-1], v[r_pointer][j]))
                j += 1
        _links = len([(s, t) for s, t in links if s == t and t == 1.0])
        score = np.sum(np.array([_links, trace])) / np.sum(v.shape)
        return min(1.0, score)
