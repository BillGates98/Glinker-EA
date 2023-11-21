from compute_files import ComputeFile
import time
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

plt.style.use('seaborn-whitegrid')


class Main:

    def __init__(self, input_path='', output_path=''):
        self.input_path = input_path
        self.output_path = output_path
        self.total = 80  # 61
        self.divider = 1e6  # 1e6
        self.labelo = 'M'  # M
        self.datasets = ['0'*(3-len(str(i))) + str(i) for i in range(1, 81)]
        self.start_time = time.time()

    def save_to_csv(self, data=None, filename=''):
        df = pd.DataFrame(data)
        df.to_csv(filename)
        return None

    def plot_data(self, data=[], datasets=[], metric=''):
        plt.figure(figsize=(4.3, 2))
        dimensions = ['Glinker']
        values = np.array(data)
        n = len(values)
        w = .3
        x = np.arange(0, len(datasets))
        # '#00b386']  # , '#ffc107', '#aa80ff']
        colors = ['#4472c4']  # '#ffc107']
        for i, value in enumerate(values):
            position = x + (w*(1-n)/2) + i*w
            plt.bar(position, value, width=w,
                    label=f'{dimensions[i]}', color=colors[i])

        plt.xticks(x, [i+1 for i in range(len(datasets))])

        plt.ylabel(metric + ' IIMB small')
        plt.ylim((0, 1))
        # plt.axhline(y=0.5, color='blue', linestyle='--')
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        file_name = self.output_path + '_histo_metrics_' + metric.lower() + '.png'
        plt.savefig(file_name)
        return None

    def plot_curve_with_points(self, x, y1, y2, y3, label1, label2, label3, means=[]):
        plt.figure(figsize=(5, 2))
        colors = ['#007acc', '#00b386', '#ff6b6b']  # , '#ffc107', '#aa80ff']
        plt.plot(x, y1, marker='.', label=label1,
                 color=colors[0], linewidth=0.5)
        plt.plot(x, y2, marker='.', label=label2,
                 color=colors[1], linewidth=0.5)
        plt.plot(x, y3, marker='.', label=label3,
                 color=colors[2], linewidth=0.5)
        plt.axhline(y=means[0], color=colors[0], linestyle='--')
        plt.axhline(y=means[1], color=colors[1], linestyle='--')
        plt.axhline(y=means[2], color=colors[2], linestyle='--')
        plt.title("Metrics on " + str(self.total) + " datasets : Pm = " +
                  str(means[0]) + ", Rm = " + str(means[1]) + " F1m  = " + str(means[2]))
        plt.legend()
        plt.grid(True)
        file_name = self.output_path + '_curve_metrics.png'
        plt.savefig(file_name)

    def plot_curve_time_with_points(self, x, y, label, mean=0.0):
        plt.figure(figsize=(5, 2))
        plt.plot(x, y, marker='.', label=label, color='skyblue', linewidth=0.5)
        plt.axhline(y=mean, color='skyblue', linestyle='--')
        plt.title("Times on " + str(self.total) +
                  " datasets : Tm = " + str(mean))
        plt.legend()
        plt.grid(True)
        file_name = self.output_path + '_curve_times.png'
        plt.savefig(file_name)

    def plot_curve_rr_with_points(self, x, y, label, mean=0.0):
        plt.figure(figsize=(5, 2))
        plt.plot(x, y, marker='.', label=label, color='skyblue', linewidth=0.5)
        plt.axhline(y=mean, color='skyblue', linestyle='--')
        plt.title("Reduction Radio on " + str(self.total) +
                  " datasets : Rrm = " + str(mean))
        plt.legend()
        plt.grid(True)
        file_name = self.output_path + '_curve_reduction_ratio.png'
        plt.savefig(file_name)

    def plot_curve_candidates_with_points(self, x, y=[], label=[], means=[]):
        plt.figure(figsize=(5, 2))
        plt.plot(x, y[0], marker='.', label=label[0],
                 color='skyblue', linewidth=0.5)
        plt.plot(x, y[1], marker='.', label=label[1],
                 color='purple', linewidth=0.5)
        plt.title("Sel. cand over pairs cand : " +
                  "Msc=" + str(round(means[1]/self.divider, 0)) + self.labelo + " vs Mpc=" + str(round(means[0]/self.divider, 0)) + self.labelo)
        plt.legend()
        plt.grid(True)
        file_name = self.output_path + '_curve_candidates_pairs.png'
        plt.savefig(file_name)
        # plt.show()

    def read_csv(self, file=''):
        df = pd.read_csv(file, index_col=None)
        return df

    def ntil(self, data=[]):
        datasets = []
        outputs = {
            'p0': [],
            'r0': [],
            'f0': [],
        }
        for i in range(int(len(data[0])/10) + 1):
            datasets.append(i+1)
            _p = data[0][i*10:(i+1)*10]
            _r = data[1][i*10:(i+1)*10]
            _f = data[2][i*10:(i+1)*10]
            if len(_p) == 0:
                outputs['p0'].append(0.0)
                outputs['r0'].append(0.0)
                outputs['f0'].append(0.0)
            else:
                outputs['p0'].append(round(np.mean(_p), 2))
                outputs['r0'].append(round(np.mean(_r), 2))
                outputs['f0'].append(round(np.mean(_f), 2))
        filename = self.output_path + 'to_glinker.csv'
        # codifile = self.output_path + 'to_codi.csv'
        # df = self.read_csv(file=codifile)
        # for i in range(len(df)):
        #     outputs['p1'].append(df.loc[i]['p'])
        #     outputs['r1'].append(df.loc[i]['r'])
        #     outputs['f1'].append(df.loc[i]['f'])
        print(np.mean([i for i in outputs['p0'] if i > 0.0]))
        print(np.mean([i for i in outputs['r0'] if i > 0.0]))
        print(np.mean([i for i in outputs['f0'] if i > 0.0]))
        # exit()
        self.plot_data(data=[outputs['p0']],
                       datasets=datasets, metric='Precision')
        self.plot_data(data=[outputs['r0']],
                       datasets=datasets, metric='Recall')
        self.plot_data(data=[outputs['f0']],
                       datasets=datasets, metric='F-measure')

        self.save_to_csv(data=outputs, filename=filename)

    def run(self):
        output = {}
        data = []
        tmp = {
            'precision': [],
            'recall': [],
            'f1score': [],
            'time': [],
            'candp': [],
            'selp': [],
            'rr': [],
        }
        print('Curve generation started 0%')
        files = ComputeFile(input_path=self.input_path,
                            output_path=self.output_path).build_list_files()
        files = [file for file in files if file.endswith('.csv')]
        indexes = self.datasets
        _datasets = []
        for index in indexes:
            output[index] = [0.0, 0.0, 0.0]
            _file = ''
            for file in files:
                if index in file:
                    _file = file
                    _datasets.append(index)
                    break
            if len(_file) > 0:
                df = self.read_csv(file=_file)
                idx_max_line = df['F1-score'].idxmax()
                ligne_max = df.loc[idx_max_line]
                output[index][0] = ligne_max['Precision']
                output[index][1] = ligne_max['Recall']
                output[index][2] = ligne_max['F1-score']

                tmp['precision'].append(ligne_max['Precision'])
                tmp['recall'].append(ligne_max['Recall'])
                tmp['f1score'].append(ligne_max['F1-score'])
                tmp['time'].append(ligne_max['RunningTime'])
                tmp['candp'].append(ligne_max['CandidatesPairs'])
                tmp['selp'].append(ligne_max['SelectedCandidates'])
                tmp['rr'].append(
                    round(1 - (ligne_max['SelectedCandidates']/ligne_max['CandidatesPairs']), 2))
                # print(ligne_max['F1-score'])
        data.append(tmp['precision'])
        data.append(tmp['recall'])
        data.append(tmp['f1score'])
        self.ntil(data=[tmp['precision'], tmp['recall'], tmp['f1score']])
        # self.plot_data(data=data)
        # exit()
        means = [round(np.mean(tmp['precision']), 2), round(np.mean(
            tmp['recall']), 2), round(np.mean(tmp['f1score']), 2)]
        x = [i for i in range(1, self.total + 1)]
        self.plot_curve_with_points(x=x, y1=tmp['precision'], y2=tmp['recall'],
                                    y3=tmp['f1score'], label1='Precision', label2='Recall', label3='F-measure', means=means)

        self.plot_curve_time_with_points(
            x=x, y=tmp['time'], label='Time (s)', mean=round(np.mean(tmp['time']), 2))

        self.plot_curve_rr_with_points(
            x=x, y=tmp['rr'], label='Reduction Ratio (%)', mean=round(np.mean(tmp['rr']), 2))

        self.plot_curve_candidates_with_points(x=x, y=[tmp['candp'], tmp['selp']], label=[
                                               'Candidates pairs', 'Selected pairs'], means=[round(np.mean(tmp['candp']), 2), round(np.mean(tmp['selp']), 2)])
        # self.datasets = _datasets
        # self.plot_data(data=data)
        # exit()
        return None


if __name__ == '__main__':
    def arg_manager():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_path", type=str,
                            default="./outputs/freebase/")
        parser.add_argument("--output_path", type=str,
                            default="./outputs/freebase/_000/")
        return parser.parse_args()
    args = arg_manager()
    Main(input_path=args.input_path,
         output_path=args.output_path).run()
