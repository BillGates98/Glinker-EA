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
    def __init__(self, truth_file='', output_file='', suffix='', embedding_name='', similarity_measure=''):
        self.truth_file = truth_file
        self.alignment_file = output_file
        self.suffix = suffix
        self.embedding_name = embedding_name
        self.similarity_measure = similarity_measure
        self.step = 0.01

    def load_graph(self, file=''):
        graph = Graph()
        graph.parse(file)
        return graph

    def read_csv(self):
        data = pd.read_csv(self.alignment_file)
        # data = data.loc[(data['cosine_sim'] >= 0.96)]
        print('Selected : ', len(data))
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
        while tmp < end:
            tmp += step
            tmp = round(tmp, 2)
            output.append(tmp)
        return output

    def append_rows_to_csv(self, new_rows, measure_file):
        try:
            df = pd.read_csv(measure_file)
        except FileNotFoundError:
            df = pd.DataFrame(
                columns=['Dataset', 'EmbeddingName', 'SimilarityMeasure', 'Precision', 'Recall', 'F1-score', 'Alpha', 'Beta'])

        new_data = pd.DataFrame(
            new_rows, columns=['Dataset', 'EmbeddingName', 'SimilarityMeasure', 'Precision', 'Recall', 'F1-score', 'Alpha', 'Beta'])
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv(measure_file, index=False)

    def orchestrator(self):
        output = {
            "recall": [0.0],
            "f1score": [0.0],
            "alpha": [0.0],
            "beta": [0.0],
            "precision": [0.0]
        }
        data = self.read_csv()
        truths = self.truths()
        alpha = 0.0  # self.step
        alphas = self.numerical_set(step=0.05)
        betas = self.numerical_set(step=0.05)
        for i in tqdm(range(len(alphas))):
            alpha = alphas[i]
            for j in range(len(betas)):
                beta = betas[j]
                # print(alpha, beta)
                if beta > alpha:
                    tmp_alignments = []
                    tmp = data.loc[(data['cosine_sim'] >= alpha)
                                   & (data['cosine_sim'] <= beta)]
                    tmp = tmp.loc[(tmp['string_similarity'] >= alpha) & (
                        tmp['string_similarity'] <= beta)]
                    # tmp = data.loc[(data['cosine_sim'] >= alpha)]
                    # tmp = tmp.loc[(tmp['string_similarity'] >= beta)]
                    if len(tmp) > 0:
                        for index in tmp.index:
                            source = str(tmp['source'][index])
                            target = str(tmp['target'][index])
                            if validators.url(source) and validators.url(target):
                                tmp_alignments.append(
                                    (source, target))
                                precision, recall, f1score = self.compute_scores(
                                    alignments=tmp_alignments, truths=truths)
                                output['recall'].append(recall)
                                output['f1score'].append(f1score)
                                output['alpha'].append(alpha)
                                output['beta'].append(beta)
                                # output['_alpha'].append(_alpha)
                                # output['_beta'].append(_beta)
                                output['precision'].append(precision)
        f1scores = np.array(output['f1score'])
        f1score_max_index = np.argmax(f1scores)
        # print(output)
        tmp = {
            'recall': output['recall'][f1score_max_index],
            'alpha': round(output['alpha'][f1score_max_index], 2),
            'beta': round(output['beta'][f1score_max_index], 2),
            'precision': output['precision'][f1score_max_index],
            'f1-score': output['f1score'][f1score_max_index]
        }
        print(tmp)
        metrics_file = './outputs/optimal_metrics_without_ablation.csv'
        new_data = [self.suffix, self.embedding_name, self.similarity_measure,
                    tmp['precision'], tmp['recall'], tmp['f1-score'], tmp['alpha'], tmp['beta']]
        self.append_rows_to_csv([new_data], metrics_file)
        return output

    def run(self):
        output = self.orchestrator()
        # self.draw_data(data=output)
        # print(output)
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
        parser.add_argument("--embedding_name", type=str, default="r2v")
        parser.add_argument("--similarity_measure",
                            type=str, default="hpp_sim")
        return parser.parse_args()

    start = time.time()
    args = arg_manager()
    truth_file = detect_file(path=args.input_path+args.suffix, type='same_as')
    output_file = detect_file(
        path='./outputs/'+args.suffix, type=args.embedding_name+'_' + args.similarity_measure + '_valid_same_as')
    print('Dataset : ', args.suffix, ' without ablation')
    print('Files : ', truth_file, ' >> ', output_file)
    Optimization(truth_file=truth_file, output_file=output_file,
                 suffix=args.suffix, embedding_name=args.embedding_name, similarity_measure=args.similarity_measure).run()
    print('Running Time : ', (time.time() - start), ' seconds ')
