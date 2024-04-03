from hpp_similarity import StringSimilarity


class DeepSimilarity:

    def __init__(self, code=''):
        # print('Deep String Similarity')
        self.code = code

    def hpp_sim(self, value1='', value2=''):
        if len(value1.lstrip()) > 0 and len(value2.lstrip()) > 0:
            return StringSimilarity(source=value1, target=value2).run()
        return 0.0

    def jaro_similarity(self, value1='', value2=''):
        def jaro_winkler_distance(s1, s2):
            len_s1 = len(s1)
            len_s2 = len(s2)

            max_len = max(len_s1, len_s2)

            match_distance = (max_len // 2) - 1

            s1_matches = [False] * len_s1
            s2_matches = [False] * len_s2

            matches = 0

            for i in range(len_s1):
                start = max(0, i - match_distance)
                end = min(i + match_distance + 1, len_s2)

                for j in range(start, end):
                    if not s2_matches[j] and s1[i] == s2[j]:
                        s1_matches[i] = True
                        s2_matches[j] = True
                        matches += 1
                        break

            if matches == 0:
                return 0.0

            transpositions = 0
            k = 0
            for i in range(len_s1):
                if s1_matches[i]:
                    while not s2_matches[k]:
                        k += 1
                    if s1[i] != s2[k]:
                        transpositions += 1
                    k += 1

            jaro_similarity = (matches / len_s1 + matches / len_s2 +
                               (matches - transpositions / 2) / matches) / 3.0

            return jaro_similarity

        jaro_distance = jaro_winkler_distance(value1, value2)  # 1.0 -
        return jaro_distance

    def hamming(self, value1='', value2=''):
        _lvalue1 = len(value1)
        _lvalue2 = len(value2)
        sim = 0
        for i in range(min(_lvalue1, _lvalue2)):
            sim += 1 if value1[i] == value2[i] else 0
        final = (sim + abs(_lvalue1 - _lvalue2)) / max(_lvalue1, _lvalue2)
        return final

    def run(self, value1='', value2='', measure=''):
        if measure == 'hpp_sim':
            return self.hpp_sim(value1=value1, value2=value2)
        elif measure == 'jaro_winkler':
            return self.jaro_similarity(value1=value1, value2=value2)
