from gensim.models import Word2Vec
from rdflib import Graph, URIRef, Namespace, Literal
from rdflib.namespace import OWL
from rdflib import Graph
import networkx as nx
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
# from node2vec import Node2Vec
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import pandas as pd
from sklearn.decomposition import PCA


class Embedding:

    def __init__(self, file='', dimension=10, model_name=''):
        self.file = file
        self.dimension = dimension
        self.model_name = model_name

    def load_file(self):
        g = Graph()
        g.parse(self.file)
        return g

    def build_triples(self, with_predicate=True):
        # g = Graph()
        # g.parse(self.file)
        triples = list(self.load_file())
        if not with_predicate:
            return [[str(triple[0]), str(triple[2])] for triple in triples]
        return [[str(triple[0]), str(triple[1]), str(triple[2])] for triple in triples]

    def build_triple_from_csv(self):
        subjects = []
        predicates = []
        objects = []
        triples = list(self.load_file())
        for triple in triples:
            subjects.append(str(triple[0]))
            predicates.append(str(triple[1]))
            objects.append(str(triple[2]))
        df = pd.DataFrame(
            {'subject': subjects, 'predicate': predicates, 'object': objects})
        # df.to_csv('tmp.tsv', sep='\t', index=False, header=False)
        return df[['subject', 'predicate', 'object']].values

    def build_w2v_model(self):
        model = Word2Vec(self.build_triples(
        ), vector_size=self.dimension, window=5, sg=1, min_count=1, workers=6)
        return model  # vector = model.wv['http://example.org/resource1']

    def build_n2v_model(self):
        nx_graph = rdflib_to_networkx_multidigraph(self.load_file())
        node2vec = Node2Vec(nx_graph, dimensions=self.dimension,
                            walk_length=30, num_walks=10, workers=5)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        return model  # model.wv['subject']

    def build_pykeen_pipeline(self, model=''):
        output = {}

        triples_factory = TriplesFactory.from_labeled_triples(
            triples=self.build_triple_from_csv(),
        )
        training = triples_factory
        validation = triples_factory
        testing = triples_factory
        d = training
        id_to_entity = {v: k for k, v in d.entity_to_id.items()}
        id_to_relation = {v: k for k, v in d.relation_to_id.items()}

        result = pipeline(
            model=model,
            loss="softplus",
            training=training,
            testing=testing,
            validation=validation,
            model_kwargs=dict(embedding_dim=self.dimension),
            optimizer_kwargs=dict(lr=0.1),
            training_kwargs=dict(num_epochs=10, use_tqdm_batch=False),
        )
        model = result.model
        entity_embeddings = model.entity_representations[0](
            indices=None).detach().cpu().numpy()
        for i, entity in enumerate(triples_factory.entity_id_to_label):
            output[id_to_entity[entity]] = entity_embeddings[i]
        # print(output)
        return output

    def run(self):
        if self.model_name == 'r2v':
            return self.build_w2v_model()
        elif self.model_name == 'n2v':
            return self.build_n2v_model()
        else:
            return self.build_pykeen_pipeline(model=self.model_name)
