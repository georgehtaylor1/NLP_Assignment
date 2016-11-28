from os import listdir
from os.path import isfile, join
import nltk
from nltk.corpus import ieer
from nltk.corpus import names as nltknames
from SPARQLWrapper import SPARQLWrapper, JSON
import re
import csv
import math
import sys

training_path = "/home/george/nltk_data/corpora/assignment/wsj_training/wsj_training/"
untagged_path = "/home/george/nltk_data/corpora/assignment/wsj_untagged/wsj_untagged/"
corpus_root = "/home/george/nltk_data/corpora/assignment/wsj_training/"
dbpedia_path_ttl = "/home/george/PycharmProjects/nlp_assignment/wrd_instances.ttl"
dbpedia_path_csv = "/home/george/PycharmProjects/nlp_assignment/wrd_instances.csv"
names = set().union(nltknames.words("male.txt"), nltknames.words("female.txt"))
titles = ["Mr", "Mrs", "Dr", "Sir", "Prof", "Professor", "Ms"]


def get_entities():
    onlyfiles = [f for f in listdir(untagged_path) if isfile(join(untagged_path, f))]
    onlyfiles.remove(".DS_Store")

    text = ""
    for f in onlyfiles:
        with open(untagged_path + f, 'r') as mf:
            text += mf.read()
    print("Files Loaded")

    sentences = nltk.sent_tokenize(text)[:100]
    print("Tokenized sentences")
    sentences2 = [nltk.word_tokenize(sent) for sent in sentences]
    print("Tokenized words")
    sentences3 = [nltk.pos_tag(sent) for sent in sentences2]
    print("POS tagged")

    zipsents = [map(list, zip(*sent)) for sent in sentences3]

    NNPs = []
    acc = []
    for j in zipsents:
        for i in range(0, len(j[0])):
            if j[1][i] == 'NNP' or j[1][i] == 'NNPS':
                acc += [j[0][i]]
            elif j[1][i] == 'CC' and len(acc) > 0:
                acc += [j[0][i]]
            else:
                if not acc == []:
                    NNPs += [" ".join(acc)]
                acc = []
    r = set(NNPs)
    print("{} entities detected".format(len(r)))
    return r


# Same as above but use a grammar and parser to extract the entities
def get_entities_grammar(grammar):
    onlyfiles = [f for f in listdir(untagged_path) if isfile(join(untagged_path, f))]
    onlyfiles.remove(".DS_Store")

    text = ""
    for f in onlyfiles:
        with open(untagged_path + f, 'r') as mf:
            text += mf.read()
    print("Files Loaded")

    sentences = nltk.sent_tokenize(text)[:1000]
    print("Tokenized sentences")
    sentences2 = [nltk.word_tokenize(sent) for sent in sentences]
    print("Tokenized words")
    sentences3 = [nltk.pos_tag(sent) for sent in sentences2]
    print("POS tagged")

    zipsents = [map(list, zip(*sent)) for sent in sentences3]

    if grammar == "": grammar = "NE: {(<NNP|NNPS>+<CC>*)+} # NAMED ENTITY"
    #grammar = """NE: {<NNP><NNP><:><VBD>}
    #                 {<)><NNP>}"""
    cp = nltk.RegexpParser(grammar)

    entities = []
    for s in sentences3:
        t = cp.parse(s)
        for st in t.subtrees():
            if st.label() == "NE":
                entities += [st]

    return entities


def get_relations():
    files = ['APW_19980314', 'APW_19980429', 'NYT_19980403', 'APW_19980424', 'NYT_19980315', 'NYT_19980407']
    for f in files:
        docs = ieer.parsed_docs(f)
        entities = []
        for d in docs:
            for s in d.text:
                if type(s) == nltk.tree.Tree:
                    entities += [(" ".join(s.leaves()), s.label())]
    print("{} relations discovered".format(len(entities)))
    return entities


def get_relation(relations, i):
    #    for (v, r) in relations:
    #       if v == i: return r
    #    return None

    hi = len(relations) + 1
    lo = 0
    mid = int(math.floor((hi + lo) / 2))
    while mid != lo:
        # print("{}, {}, {}".format(lo,mid,hi))
        if i > relations[mid][0]:
            lo = mid
        else:
            hi = mid
        mid = int(math.floor((hi + lo) / 2))
    if relations[mid + 1][0] == i:
        return relations[mid + 1][1]
    if relations[mid][0] == i:
        return relations[mid][1]
    return None


def relate_entities(relations, entities):
    rs = []
    fails = []
    i = 0
    for e in entities:
        r = get_relation(relations, e)
        if not (r is None):
            rs += [(e, r)]
        elif isName(e):
            rs += [(e, "Person")]
        else:
            fails += [e]
        i += 1
        if i % 10 == 0: print "\r{}".format(i)
    print("{} entities related out of {} entities with {} relations provided"
          .format(len(rs), len(entities), len(relations)))
    return rs, fails


def relate_entities_sparql(entities):
    rs = []
    rejections = []
    i = 0
    for e in entities:
        r = get_relation_sparql(e)
        if not (r is None):
            rs += [(e, r)]
        elif isName(e):
            rs += [(e, "PERSON")]
        else:
            rejections += [e]
        i += 1
        sys.stdout.write("\r%d%%" % int(i * 100 / len(entities)))
        sys.stdout.flush()
    print("\n{} entities related out of {} entities"
          .format(len(rs), len(entities)))
    return rs, rejections


def isName(entity):
    name = entity.split()
    return all((n in names) or (re.match(r'\w+\.')) for n in name)


def get_relations_dbpedia():
    relations = None
    with open(dbpedia_path_csv) as f:
        relations = [tuple(line) for line in csv.reader(f)]
    print("{} relations loaded".format(len(relations)))
    return sorted(relations)


def get_relation_sparql(entity):
    es = entity.split()
    es = [w for w in es if not w in titles]
    entity = "_".join(es)
    entity = re.sub(r'[^\P{P}\w\.\_]+', "", entity)

    sp = SPARQLWrapper("http://dbpedia.org/sparql")
    sp.setQuery(""" select ?t
                    where {
                        OPTIONAL { <http://dbpedia.org/resource/%s> a ?t } .
                    }""" % entity)
    sp.setReturnFormat(JSON)
    try:
        results = sp.query().convert()
        if results["results"]["bindings"] == [{}]: return None

        for r in results["results"]["bindings"]:
            v = r["t"]["value"]
            if v == "http://dbpedia.org/ontology/Person":
                return "PERSON"
            elif v == "http://dbpedia.org/ontology/Organisation":
                return "ORGANISATION"
            elif v == "http://dbpedia.org/ontology/Location":
                return "LOCATION"
    except:
        pass

    return None


def convert_dbpedia():
    r = []
    reg_name = re.compile("/resource/\S+>", re.UNICODE)
    reg_ont = re.compile("(/ontology/\S+)|(owl#\S+)>", re.UNICODE)

    with open(dbpedia_path_ttl, 'r') as mf:
        mf.readline()
        i = 0
        for l in mf:
            try:
                name = reg_name.search(l).group()[10:-1].replace('_', ' ')
                t = reg_ont.search(l).group()
                if t[3] == '#':
                    r += [[name, t[4:-1]]]
                else:
                    r += [[name, t[10:-1]]]
            except:
                pass
    print "{} relations extracted".format(len(r))

    with open(dbpedia_path_csv, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(r)

    return r


def get_test_data():
    people = []
    locations = []
    organisations = []

    onlyfiles = [f for f in listdir(training_path) if isfile(join(training_path, f))]
    onlyfiles.remove(".DS_Store")

    text = ""
    for f in onlyfiles:
        with open(training_path + f, 'r') as mf:
            text += mf.read()
    print("Files Loaded")

    r = re.compile(r'<ENAMEX TYPE=".*?">.*?</ENAMEX>')
    rs = re.findall(r, text)
    r2 = re.compile(r'["<>]')
    rs2 = map(lambda s: r2.split(s), rs)
    rs2 = list(set(map(lambda l: (l[4], l[2]), rs2)))
    print("%d entities extracted" % len(rs2))
    return rs2


def run():
    print ""
    print("Collecting entities")
    print("-------------------")
    e = get_entities()

    # print ""
    # print("Collecting relations")
    # print("--------------------")
    # r = get_relations_dbpedia()

    print ""
    print("Relating entities")
    print("-----------------")
    # er, fails = relate_entities(r, e)
    er, fails = relate_entities_sparql(e)

    print ""
    print("Loading Test Data")
    print("-----------------")
    t = get_test_data()

    print("COMPLETE")
    return er, fails
