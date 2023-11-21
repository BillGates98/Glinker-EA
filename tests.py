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

vocabulary = dict()
synonyms = dict()

ds = DeepSimilarity(code='*')

output_alignements = {}

already_treated = {}
#  and not sub1 in output_alignements


def append_rows_to_csv(new_rows, measure_file):
    try:
        df = pd.read_csv(measure_file)
    except FileNotFoundError:
        df = pd.DataFrame(
            columns=['Dataset', 'Precision', 'Recall', 'F1-score', 'Threshold', 'CandidatesPairs', 'SelectedCandidates', 'RunningTime'])

    new_data = pd.DataFrame(
        new_rows, columns=['Dataset', 'Precision', 'Recall', 'F1-score', 'Threshold', 'CandidatesPairs', 'SelectedCandidates', 'RunningTime'])
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(measure_file, index=False)


def calculate_alignment_metrics(output_file, truth_file, suffix, threshold, countpairs, selectedcount, runningtime):
    measure_file = output_file.replace(
        'tmp_valid_same_as.ttl', 'measure_file.csv')
    output_graph = Graph()
    output_graph.parse(output_file, format="turtle")

    truth_graph = Graph()
    truth_graph.parse(truth_file, format="turtle")

    found_alignments = set(output_graph.subjects())
    true_alignments = set(truth_graph.subjects())
    print('Count of true alignments : ', len(true_alignments))
    intersection = len(found_alignments.intersection(true_alignments))
    precision = round(intersection /
                      len(found_alignments) if len(found_alignments) > 0 else 0.0, 2)
    recall = round(intersection /
                   len(true_alignments) if len(true_alignments) > 0 else 0.0, 2)
    f_measure = round(2 * (precision * recall) / (precision +
                                                  recall) if (precision + recall) > 0 else 0.0, 2)

    append_rows_to_csv([(suffix, precision, recall, f_measure, threshold,
                         countpairs, selectedcount, round(runningtime, 2))], measure_file)
    return {
        "precision": precision,
        "recall": recall,
        "f_measure": f_measure
    }

# End of Metrics handling


def sigmoid(value):
    return 1 / (1 + np.exp(value))


def cosine_sim(v1=[], v2=[]):
    output = 0.0
    dot = np.dot(v1, v2)
    cosine = dot / (np.linalg.norm(v1) * np.linalg.norm(v2))
    output = cosine
    # (IIMB, Person, 0.6), (Restaurant, 0.5), (doremus, 0.7), (SPIM-s, 0.7|0.8|0.9), (SPIM-l, 0.9|0.8)
    if output >= 0.5:
        return True, output
    return False, output


def sim(entity1=[], entity2=[], cosim=0.0, threshold=0.0):
    jaros = []
    for p1, o1 in entity1:
        if not validators.url(o1):
            tmp = []
            for p2, o2 in entity2:
                if not validators.url(o2):
                    jaro_sim = round(ds.jaro_similarity(
                        value1=o1, value2=o2), 2)
                    if jaro_sim > 0:
                        tmp.append(jaro_sim)
            if len(tmp) > 0:
                jaros.append(max(tmp))
    leno = len(jaros)
    if leno > 1:
        _mean = round((np.mean(np.array(jaros))), 2)  # / cosim
        if _mean >= 0.5:
            decision = round((_mean * leno)/cosim, 2)
            decision = round(1-sigmoid(decision), 2)
            if decision >= threshold:  # 0.22
                # print('Decision taked for value : ', decision, ' old value : ',
                #       cosim, ' _mean : ', _mean, ' size : ', len(jaros))
                return True
    return False


def create_and_save_rdf_from_dict(input_dict, output_file):
    graph = Graph()
    # owl = Namespace("http://www.w3.org/2002/07/owl#")
    for source, target in input_dict.items():
        source_uri = URIRef(source)
        target_uri = URIRef(target)
        graph.add((source_uri, OWL.sameAs, target_uri))
    graph.serialize(destination=output_file, format="turtle")


def get_rdf_subjects(rdf_graph):
    output = list(rdf_graph.subjects())
    return output


def get_rdf_triples(rdf_graph):
    subjects = {}
    objects = {}
    for s, p, o in tqdm(rdf_graph):
        s = str(s)
        p = str(p)
        o = str(o)
        if not s in subjects:
            subjects[s] = []

        if not o in objects:
            objects[o] = 0

        objects[o] += 1
        subjects[s].append((p, o))
    return subjects, objects


def random_selections(data={}, k=0):
    entities = list(data.keys())
    output = {}
    _randed = random.choices(entities, k=int(len(data)*k))
    for e in _randed:
        output[e] = data[e]
    return output


# End of embedding functions
#

def parallel_running(sub1, sub2, vector1, vector2, subs1, subs2, threshold):

    v, cos = cosine_sim(v1=vector1, v2=vector2)
    if v:
        if sim(entity1=subs1[sub1], entity2=subs2[sub2], cosim=cos, threshold=threshold):
            # already_treated[sub1] = sub2
            return sub1, sub2, 1
    return None, None, 0


dimension = 20
# algo = 'w2v'
algo = 'r2v'


def process_rdf_files(file1, file2, output_file, truth_file, suffix, threshold):

    graph1 = Graph()
    graph1.parse(file1)
    print('Source file loaded ..100%')

    graph2 = Graph()
    graph2.parse(file2)
    print('Target file loaded ..100%')

    source_embeddings = Embedding(
        file=file1, dimension=dimension, algo=algo).run()
    target_embeddings = Embedding(
        file=file2, dimension=dimension, algo=algo).run()
    valid_alignements = Embedding(
        file=truth_file, dimension=dimension).build_triples(with_predicate=False)

    graph3 = Graph()
    graph3.parse(truth_file)
    print('Truth file loaded ..100%')

    print('Graph1 Subjects\'s and Objects\' list are building ..0%')
    subjects1, objects1 = get_rdf_triples(graph1)
    print('Graph2 Subjects\'s and Objects\' list are building ..0%')
    subjects2, objects2 = get_rdf_triples(graph2)
    print('Building ended')

    # print('Candidates reducing ')
    print('Instances of source : ', len(list(subjects1.keys())))
    print('Instances of target : ', len(list(subjects2.keys())))
    # subjects1 = random_selections(subjects1, k=1.0)
    # subjects2 = random_selections(subjects2, k=0.01)
    # print('Instances of source : ', len(list(subjects1.keys())))
    # print('Instances of target : ', len(list(subjects2.keys())))

    # exit
    # for s, _, t in graph3:
    #     _, val = cosine_sim(v1=source_embeddings.wv[str(
    #         s)], v2=target_embeddings.wv[str(t)])
    #     if sim(entity1=subjects1[str(s)], entity2=subjects2[str(t)], cosim=val):
    #         output_alignements[str(s)] = str(t)
    pairs = list(itertools.product(
        list(subjects1.keys()), list(subjects2.keys())))
    # print(f'{len(pairs)} total candidates pairs ')
    print('In all : ', len(pairs))
    _pairs = []
    # for sub1 in tqdm(subjects1):
    #     _subjects2 = subjects2  # random_selections(subjects2, k=0.2)
    #     for sub2 in _subjects2:
    #         inter = set(subjects1[sub1]) & set(subjects2[sub2])
    #         # (IIMB, person, restaurant, 0), (doremus, 1), (Spim-s, Spim-l, 1),
    #         # v, _ = cosine_sim(
    #         # v1=source_embeddings.wv[sub1], v2=target_embeddings.wv[sub2])
    #         if len(inter) >= 0:
    #             _pairs.append((sub1, sub2))
    print('they are to compute : ', len(_pairs))
    # exit()
    count = 0
    with multiprocessing.Pool(processes=13) as pool:
        results = pool.starmap(parallel_running,
                               [(sub1, sub2, source_embeddings.wv[sub1], target_embeddings.wv[sub2], subjects1, subjects2, threshold)
                                for sub1, sub2 in tqdm(_pairs) if sub1 in source_embeddings.wv and sub2 in target_embeddings.wv])
        for sub1, sub2, status in results:
            if sub1 != None and sub2 != None:
                output_alignements[sub1] = sub2
                count = count + 1
    print('All are : ', len(results), ' count : ', count)
    print(f' \nThey are {len(list(output_alignements.keys()))} in all')
    create_and_save_rdf_from_dict(output_alignements, output_file)
    end_time = time.time()
    execution_time = end_time - start_time
    metrics = calculate_alignment_metrics(
        output_file, truth_file, suffix, threshold, len(pairs), len(_pairs), execution_time)
    print("Precision : ", metrics["precision"])
    print("Recall : ", metrics["recall"])
    print("F-measure : ", metrics["f_measure"])
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
        parser.add_argument("--suffix", type=str, default="spaten_hobbit")
        parser.add_argument("--threshold", type=float, default=0.0)
        return parser.parse_args()
    args = arg_manager()
    file1 = detect_file(path=args.input_path+args.suffix, type='source')
    file2 = detect_file(path=args.input_path+args.suffix, type='target')
    truth_file = detect_file(path=args.input_path +
                             args.suffix, type='valid_same_as')
    output_path = args.output_path + args.suffix
    output_file = output_path + '/tmp_valid_same_as.ttl'
    print(args.output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print(file1, file2, output_file, truth_file)
    process_rdf_files(file1, file2, output_file, truth_file,
                      args.suffix, args.threshold)
