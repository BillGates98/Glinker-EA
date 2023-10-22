from gensim.models import Word2Vec
from rdflib import Graph, URIRef, Namespace, Literal
from rdflib.namespace import OWL
from rdflib import Graph
import networkx as nx
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
from node2vec import Node2Vec


class Embedding:

    def __init__(self, file='', dimension=10, algo=''):
        self.file = file
        self.dimension = dimension
        self.algo = algo

    def load_file(self):
        g = Graph()
        g.parse(self.file)
        return g
    
    def build_triples(self, with_predicate=True):
        g = Graph()
        g.parse(self.file)
        triples = list(g)
        if not with_predicate :
            return [[str(triple[0]), str(triple[2])] for triple in triples]
        return  [[str(triple[0]), str(triple[1]), str(triple[2])] for triple in triples]
    
    def build_w2v_model(self):
        model = Word2Vec(self.build_triples(), vector_size=self.dimension, window=5, sg=1, min_count=1, workers=6)
        return model # vector = model.wv['http://example.org/resource1']
    
    def build_n2v_model(self):
        nx_graph = rdflib_to_networkx_multidigraph(self.load_file())
        node2vec = Node2Vec(nx_graph, dimensions=self.dimension, walk_length=30, num_walks=20, workers=5)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)  
        return model # model.wv['subject']

    def run(self):
        if self.algo == 'w2v' :
            return self.build_w2v_model()
        elif self.algo == 'n2v' :
            return self.build_n2v_model()


    