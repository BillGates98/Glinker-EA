import os
from compute_files import ComputeFile
import time
import argparse
from embeddings import Embedding
import itertools
import pprint
import random
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

class FeatureBuilding: 

    def __init__(self, input_path='', output_path='', suffix='', algo='', dimension=10):
        self.input_path = input_path + suffix + '/'
        self.output_path = output_path
        self.suffix = suffix
        self.dimension = dimension
        self.algo = algo
        files = ComputeFile(input_path=self.input_path).build_list_files()
        self.source_file = self.filter(keyword='source', all=files)
        self.target_file = self.filter(keyword='target', all=files)
        self.valid_file = self.filter(keyword='valid_same_as', all=files)
        self.output_path = output_path
        self.outputs = {
            'source_id': [],
            'target_id': [],
            'pair_id': [],
            'label': []
        }
        print("Processing started for " , self.suffix,  "-", dimension)
        self.start_time = time.time()
    
    def filter(self, keyword='', all=[]):
        return [file for file in all if keyword in file][0]
    
    def prepare_features(self):
        for i in range(self.dimension):
            self.outputs['feature_'+str(i)] = []
        return None
    
    def balance_sample(self, df, label_col):
        positives = df[df[label_col] == True]
        negatives = df[df[label_col] == False]
        # min_samples = min(len(positives), len(negatives))
        
        random_state = 42
        test_size=0.3
        # positive_sample = positives.sample(n=min_samples, random_state=random_state)
        # negative_sample = negatives.sample(n=min_samples, random_state=random_state)
        
        p_train_set, p_test_set = train_test_split(positives, test_size=test_size, random_state=random_state)
        n_train_set, n_test_set = train_test_split(negatives, test_size=test_size, random_state=random_state)
        print('* dim-', self.dimension)
        print('postives :', len(positives), ' negatives :', len(negatives))
        print('postive test : ', len(p_test_set), ' negative test : ', len(n_test_set))
        print('postive train : ', len(p_train_set), ' negative train : ', len(n_train_set))
        print('* dim-', self.dimension)

        train_balanced_sample = pd.concat([p_train_set, n_train_set])
        test_balanced_sample = pd.concat([p_test_set, n_test_set])
        return test_balanced_sample, train_balanced_sample

    def save_to_csv(self, output={}):
        output_path = self.output_path + self.suffix + '-' + str(self.dimension) + '/feature_vector/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        prefix_file = output_path + ''
        df = pd.DataFrame(output)
        test_sample, train_sample = self.balance_sample(df, label_col='label')
        # differents features files
        test_sample.to_csv(prefix_file + 'feature_vector_test.csv')
        train_sample.to_csv(prefix_file + 'feature_vector_train.csv')
        df.to_csv(prefix_file + 'feature_vector.csv')

        return None

    def run(self):
        source_embeddings = Embedding(file=self.source_file, dimension=self.dimension, algo=self.algo).run()
        target_embeddings = Embedding(file=self.target_file, dimension=self.dimension, algo=self.algo).run()
        valid_alignements = Embedding(file=self.valid_file, dimension=self.dimension).build_triples(with_predicate=False)
        self.prepare_features()
        pairs = []
        sources = []
        targets = []
        similarities = []
        for source_id, target_id in valid_alignements:
            if source_id in source_embeddings.wv and target_id in target_embeddings.wv : 
                self.outputs['source_id'].append(source_id)
                self.outputs['target_id'].append(target_id)
                sources.append(source_id)
                targets.append(target_id)
                pair_id = source_id + '-' + target_id
                self.outputs['pair_id'].append(pair_id)
                pairs.append(pair_id)
                self.outputs['label'].append(True)
                for i in range(self.dimension):
                    self.outputs['feature_'+str(i)].append(source_embeddings.wv[source_id][i] + target_embeddings.wv[target_id][i])
                similarity = cosine_similarity(source_embeddings.wv[source_id].reshape(1, -1), target_embeddings.wv[target_id].reshape(1, -1))
                similarities.append(similarity)
        print('Moyenne : ', np.mean(np.array(similarities)))
        print('Min : ', np.min(np.array(similarities)))

        product = list(itertools.product(sources, targets))
        n = len(valid_alignements)
        j = 0
        already_taken = []
        similarities = []
        while( j < n):
            source_id, target_id = random.choice(product)
            pair_id = source_id + '-' + target_id
            if pair_id in already_taken :
                continue
            if source_id in source_embeddings.wv and target_id in target_embeddings.wv : 
                if not pair_id in pairs :
                    self.outputs['source_id'].append(source_id)
                    self.outputs['target_id'].append(target_id)
                    self.outputs['pair_id'].append(pair_id)
                    self.outputs['label'].append(False)
                    already_taken.append(pair_id)
                    for i in range(self.dimension):
                        self.outputs['feature_'+str(i)].append(source_embeddings.wv[source_id][i] + target_embeddings.wv[target_id][i])
                    j = j + 1
                similarity = cosine_similarity(source_embeddings.wv[source_id].reshape(1, -1), target_embeddings.wv[target_id].reshape(1, -1))
                similarities.append(similarity)
                
        # print(similarities)
        print('Moyenne : ', np.mean(np.array(similarities)))
        print('Min : ', np.min(np.array(similarities)))
        self.save_to_csv(output=self.outputs)
        print("Processing ended for " , self.suffix, "-", self.dimension)

        return None


if __name__ == '__main__' :
    def arg_manager():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_path", type=str, default="./data/")
        parser.add_argument("--output_path", type=str, default="./outputs/")
        parser.add_argument("--suffix", type=str, default="doremus")
        parser.add_argument("--algo", type=str, default="w2v")
        parser.add_argument("--dimension", type=int, default=10)
        return parser.parse_args()
    args = arg_manager()
    FeatureBuilding(input_path=args.input_path, output_path=args.output_path, suffix=args.suffix, dimension=args.dimension, algo=args.algo).run()