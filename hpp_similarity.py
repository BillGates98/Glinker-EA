import numpy as np
import string
import argparse


class StringSimilarity:

    def __init__(self, source='', target=''):
        self.symbols = [] if len(
            list(string.printable)) == 0 else list(string.printable)
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
        # print(v)
        tmp = []
        a = int(np.sum(v.shape)/2)
        for k in range(-a, a, 1):
            tmp.append((np.sum(np.diag(v, k=k)) /
                        np.mean(v.shape)))
        tmp = np.array(tmp)
        tmp[::-1].sort()
        result = 0.0  # max(tmp)
        for a in range(0, len(tmp)):
            result += tmp[a]/(a+1)
        # correction
        if result > 1.0:
            decimal_part = result % 1
            result = result - decimal_part*(1+decimal_part)
            result = min(result, 1.0)
        # print('Result : ', result)
        return max(result, max(tmp))


if __name__ == "__main__":
    def arg_manager():
        parser = argparse.ArgumentParser()
        parser.add_argument("--value1", type=str, default="Conference")
        parser.add_argument("--value2", type=str, default="confarence")
        return parser.parse_args()
    args = arg_manager()
    value1 = args.value1
    value2 = args.value2
    score = StringSimilarity(source=value1, target=value2).run()
    print('The similarity score between \'', value1.lstrip(),
          '\' and \'', value2.lstrip(), '\' is ', score, '.')
