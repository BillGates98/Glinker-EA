from strsimpy.jaro_winkler import JaroWinkler
from hpp_similarity import StringSimilarity
import time


class SimilarityMeasure:

    def __init__(self, source='', target=''):
        self.source = source
        self.target = target

    def jaro_winkler(self):
        start = time.time()
        jarowinkler = JaroWinkler()
        value = jarowinkler.similarity(self.source, self.target)
        return {'value': value, 'time': time.time()-start}

    def hpp_sim(self):
        start = time.time()
        value = StringSimilarity(source=self.source, target=self.target).run()
        return {'value': value, 'time': time.time()-start}

    def run(self):
        output = {
            'Jaro-Winkler': self.jaro_winkler(),
            'HPP': self.hpp_sim(),
        }
        return output
