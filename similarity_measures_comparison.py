from collections import OrderedDict
from similarity_measures import SimilarityMeasure


def build_alphabet(sentences=[]):
    output = set()
    for word in sentences:
        word = word.replace(' ', '')
        tmp = list(OrderedDict.fromkeys(word).keys())
        output.update(tmp)
    return list(output)


def sim_measure_data():
    data = [
        ('Research', 'Scientist'),
        ('Research', 'Jon wick'),
        ('John wick', 'Scientist'),
        ('John wick', 'Jon wick')
    ]
    for s, t in data:
        print("Strings : <", s, "> -String- <", t, '>')
        sim = SimilarityMeasure(source=s, target=t).run()
        print(sim, '\n')


def comparison_with_jaro_data():
    data = [
        ('shackleford', 'shackleford'),
        ('shackleford', 'shackelford'),
        ('cunningham', 'cunnigham'),
        ('campell', 'campbell'),
        ('nichleson', 'nichulson'),
        ('massey', 'massie'),
        ('galloway', 'calloway'),
        ('lampley', 'campley'),
        ('frederick', 'frederic'),
        ('michele', 'michelle'),
        ('jesse', 'jessie'),
        ('marhta', 'martha'),
        ('jonathon', 'jonathan'),
        ('julies', 'juluis'),
        ('jeraldine', 'geraldine'),
        ('yvette', 'yevett'),
        ('tanya', 'tonya'),
        ('dwayne', 'duane'),
        ('JON', 'JOHN'),
        ('HARDIN', 'MARTINEZ'),
        ('ITMAN', 'SMITH'),
    ]
    for s, t in data:
        print("Strings : <", s, "> -vs- <", t, ">")
        sim = SimilarityMeasure(source=s, target=t).run()
        print(sim, '\n')


def advancing():
    data = [('jeraldien', 'geraldine')]
    s, t = data[0]
    for i in range(1, len(s)):
        tmp_s = list(s)
        tmp_t = list(t)
        j = 1
        while j <= i:
            tmp_s[j-1] = tmp_s[j]
            tmp_t[j-1] = tmp_t[j]
            j += 1
        tmp_s[j-1] = 'j'
        tmp_t[j-1] = 'g'
        data.append((''.join(tmp_s), ''.join(tmp_t)))
    for s, t in data:
        print("Strings : <", s, "> vs <", t, '>')
        sim = SimilarityMeasure(source=s, target=t).run()
        print(sim, '\n')

# Jaro winkler metric works better for single words than it does for multiple word strings.


comparison_with_jaro_data()
# advancing()
# sim_measure_data()
