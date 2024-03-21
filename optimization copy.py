from rdflib import Graph
from compute_files import ComputeFile
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import validators
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


class Optimization:
    def __init__(self, truth_file='', output_file='', suffix=''):
        self.truth_file = truth_file
        self.alignment_file = output_file
        self.suffix = suffix
        self.step = 0.01

    def load_graph(self, file=''):
        graph = Graph()
        graph.parse(file)
        return graph

    def read_csv(self):
        data = pd.read_csv(self.alignment_file)
        return data

    def truths(self):
        output = []
        graph = self.load_graph(file=self.truth_file)
        for s, _, o in graph:
            output.append((str(s), str(o)))
        return output

    def compute_scores(self, alignments=[], truths=[]):
        _alignments = set(alignments)
        _truths = set(truths)
        intersection = len(_alignments.intersection(_truths))
        precision = round(intersection /
                          len(_alignments) if len(_alignments) > 0 else 0.0, 2)
        recall = round(intersection /
                       len(_truths) if len(_truths) > 0 else 0.0, 2)
        f1score = round(2 * (precision * recall) / (precision +
                                                    recall) if (precision + recall) > 0 else 0.0, 2)
        # print(len(alignments), len(truths))

        return precision, recall, f1score

    def draw_data(self, data={}):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(data['alpha'], data['beta'], data['precision'])
        ax.set_xlabel("alpha")
        ax.set_ylabel("beta")
        ax.set_zlabel("Precision")
        plt.title('Optimization on ' + self.suffix)
        plt.savefig('./outputs/' + self.suffix + '/_figure.pdf')
        return None

    def numerical_set(self, step=0.01, end=1.0):
        output = []
        tmp = step
        while tmp < 1.0:
            tmp += step
            output.append(tmp)
        return output

    def orchestrator(self):
        output = {
            "recall": [],
            "f1score": [],
            "alpha": [],
            "beta": [],
            "_alpha": [],
            "_beta": [],
            "precision": []
        }
        data = self.read_csv()
        truths = self.truths()
        alpha = 0.0  # self.step
        alphas = self.numerical_set(step=0.1)
        betas = self.numerical_set(step=0.1)
        for i in tqdm(range(len(alphas))):
            alpha = alphas[i]
            for j in range(len(betas)):
                beta = betas[j]
                # print(alpha, beta)
                if beta > alpha:
                    tmp = data.loc[(data['cosine_sim'] >= alpha)
                                   & (data['cosine_sim'] < beta)]

                    for _i in range(len(alphas)):
                        _alpha = alphas[_i]
                        for _j in range(len(betas)):
                            _beta = betas[_j]
                            # print(alpha, beta)
                            if _beta > _alpha:
                                tmp_alignments = []
                                _tmp = tmp.loc[(tmp['string_similarity'] >= _alpha) & (
                                    tmp['string_similarity'] < _beta)]
                                if len(_tmp) > 0:
                                    for _index in _tmp.index:
                                        source = str(_tmp['source'][_index])
                                        target = str(_tmp['target'][_index])
                                        if validators.url(source) and validators.url(target):
                                            tmp_alignments.append(
                                                (source, target))
                                    precision, recall, f1score = self.compute_scores(
                                        alignments=tmp_alignments, truths=truths)
                                    output['recall'].append(recall)
                                    output['f1score'].append(f1score)
                                    output['alpha'].append(alpha)
                                    output['beta'].append(beta)
                                    output['_alpha'].append(_alpha)
                                    output['_beta'].append(_beta)
                                    output['precision'].append(precision)

        return output

    def run(self):
        output = self.orchestrator()
        self.draw_data(data=output)
        print(output)
        return None


if __name__ == "__main__":
    def detect_file(path='', type=''):
        files = ComputeFile(input_path=path).build_list_files()
        for v in files:
            if type in v:
                return v
        return None

    def arg_manager():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_path", type=str, default="./inputs/")
        parser.add_argument("--suffix", type=str, default="doremus")
        return parser.parse_args()

    start = time.time()
    args = arg_manager()
    truth_file = detect_file(path=args.input_path+args.suffix, type='same_as')
    output_file = detect_file(path='./outputs/'+args.suffix, type='same_as')
    print('Dataset : ', args.suffix)
    print('Files : ', truth_file, ' >> ', output_file)
    Optimization(truth_file=truth_file, output_file=output_file,
                 suffix=args.suffix).run()
    print('Running Time : ', (time.time() - start), ' seconds ')
