import os
from rdflib import Graph
from deep_similarity import DeepSimilarity
import numpy as np
import random
import validators
from rdflib import Graph, URIRef, Namespace, Literal
from rdflib.namespace import OWL
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
from rdflib import Graph, URIRef
from tqdm import tqdm
import time
import multiprocessing
from embeddings import Embedding
import itertools
import argparse
from compute_files import ComputeFile
import pandas as pd

start_time = time.time()


class Linking:
    def __init__(self, cpu=10, source_file='', target_file='', truth_file='', output_file='', suffix='', embedding_name='r2v', dimension=200, alpha=0.0, beta=0.0, similarity_measure=''):
        self.source_file = source_file
        self.target_file = target_file
        self.truth_file = truth_file
        self.output_file = output_file
        self.suffix = suffix
        self.dimension = dimension
        self.embedding_name = embedding_name
        self.alpha = alpha
        self.beta = beta
        self.similarity_measure = similarity_measure
        self.cpu = cpu
        self.deepSimString = DeepSimilarity(code='*')

    def load_graph(self, file=''):
        graph = Graph()
        graph.parse(file)
        return graph

    def extract_subjects(self, graph=None):
        output = {}
        for s, p, o in graph:
            if not s in output:
                output[s] = []
            output[s].append((p, o))
        return output

    def string_chain(self, entity=[]):
        output = []
        for _, o in entity:
            if not validators.url(str(o)):
                output.append(str(o))
        return output

    def compute_similarity_score(self, entity1=[], entity2=[]):
        literals1 = self.string_chain(entity=entity1)
        literals2 = self.string_chain(entity=entity2)
        sims = []
        for entity1 in literals1:
            tmp = []
            for entity2 in literals2:
                if len(entity1.lstrip()) > 0 and len(entity2.lstrip()) > 0:
                    sim = self.deepSimString.run(
                        value1=entity1, value2=entity2, measure=self.similarity_measure)
                    tmp.append(sim)
            if len(tmp) > 0:
                sims.append(max(tmp))
        if len(sims) > 0:
            return np.mean(sims)
        return 0.0

    def cosine_sim(self, v1=[], v2=[]):
        dot = np.dot(v1, v2)
        cosine = dot / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return cosine

    def sim(self, entity1=[], entity2=[]):
        output = []
        for p1, o1 in entity1:
            if not validators.url(o1):
                tmp = []
                for p2, o2 in entity2:
                    if not validators.url(o2):
                        _sim = round(self.deepSimString.bill_sim(
                            value1=o1, value2=o2), 2)
                        if _sim > 0:
                            tmp.append(_sim)
                if len(tmp) > 0:
                    output.append(max(tmp))
        leno = len(output)
        decision = 0.0
        if leno >= 1:
            decision = round((np.mean(np.array(output))), 2)
        return decision

    def create_and_save_rdf_from_dict(self, input_dict, output_file):
        graph = Graph()
        # owl = Namespace("http://www.w3.org/2002/07/owl#")
        for source, target in input_dict.items():
            source_uri = URIRef(source)
            target_uri = URIRef(target)
            graph.add((source_uri, OWL.sameAs, target_uri))
        graph.serialize(destination=output_file, format="turtle")

    def save_results(self, data={}):
        df = pd.DataFrame.from_dict(data)
        df.to_csv(self.output_file.replace('ttl', 'csv'))
        return None

    def parallel_running(self, source, s_entity, target, t_entity, cosine_sim):
        status = self.sim(entity1=s_entity, entity2=t_entity)
        return source, target, cosine_sim, status

    def process_candidates(self, candidates=[]):
        output = {'source': [], 'target': [],
                  'cosine_sim': [], 'string_similarity': []}
        with multiprocessing.Pool(processes=self.cpu) as pool:
            results = pool.starmap(self.parallel_running,
                                   [(source, s_entity, target, t_entity, cosine_sim)
                                    for source, s_entity, target, t_entity, cosine_sim in candidates])
            for s, t, cosine_sim, sim in results:
                if sim >= self.beta:
                    output['source'].append(s)
                    output['target'].append(t)
                    output['cosine_sim'].append(cosine_sim)
                    output['string_similarity'].append(sim)
        return output

    def reduce_candidate_pairs(self, embeddings={}, source_entities={}, target_entities={}):
        output = []
        _source_entities = list(source_entities.keys())
        _target_entities = list(target_entities.keys())
        for source in _source_entities:
            source_vector = embeddings[str(source)]
            for target in _target_entities:
                target_vector = embeddings[str(target)]
                cosine_sim = self.cosine_sim(
                    v1=source_vector, v2=target_vector)
                if cosine_sim >= self.alpha:
                    output.append(
                        (source, source_entities[source], target, target_entities[target], cosine_sim))
        return output

    def run(self):

        source_graph = self.load_graph(self.source_file)
        source_subjects = self.extract_subjects(source_graph)
        print('Source KG loaded ..100%')

        target_graph = self.load_graph(self.target_file)
        target_subjects = self.extract_subjects(target_graph)
        print('Target KG loaded ..100%')

        graph = source_graph + target_graph
        print('KGs merged ..100%')

        embeddings = Embedding(file=None,
                               graph=graph, dimension=self.dimension, model_name=self.embedding_name).run()

        print('Truths KG loaded ..100%')

        print('Building ended')
        candidates = self.reduce_candidate_pairs(
            embeddings=embeddings, source_entities=source_subjects, target_entities=target_subjects)

        print(">> Size of pairs candidates : ", len(candidates))
        output_alignments = self.process_candidates(candidates=candidates)
        print("## Size of final alignments : ",
              len(output_alignments['source']))
        self.save_results(data=output_alignments)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f" \n Running time : {execution_time} seconds")


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
        parser.add_argument("--output_path", type=str, default="./outputs/")
        parser.add_argument("--suffix", type=str, default="doremus")
        parser.add_argument("--cpu", type=int, default=13)
        parser.add_argument("--embedding_name", type=str, default="r2v")
        parser.add_argument("--dimension", type=int, default=300)
        parser.add_argument("--alpha", type=float, default=0.95)
        parser.add_argument("--beta", type=float, default=0.9)
        parser.add_argument("--similarity_measure",
                            type=str, default="bill_sim")
        return parser.parse_args()
    args = arg_manager()
    source_file = detect_file(path=args.input_path+args.suffix, type='source')
    target_file = detect_file(path=args.input_path+args.suffix, type='target')
    truth_file = detect_file(path=args.input_path +
                             args.suffix, type='valid_same_as')
    output_path = args.output_path + args.suffix
    output_file = output_path + '/tmp_' + args.embedding_name + \
        '_' + args.similarity_measure + '_valid_same_as.ttl'
    # print(args.output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print('Files :> \n ', source_file, target_file, output_file, truth_file)

    Linking(cpu=args.cpu, source_file=source_file, target_file=target_file, truth_file=truth_file, output_file=output_file, suffix=args.suffix,
            embedding_name=args.embedding_name, dimension=args.dimension, alpha=args.alpha, beta=args.beta, similarity_measure=args.similarity_measure).run()
