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
from collections import defaultdict

training_path = "/home/george/nltk_data/corpora/assignment/wsj_training/wsj_training/"
untagged_path = "/home/george/nltk_data/corpora/assignment/wsj_untagged/wsj_untagged/"
corpus_root = "/home/george/nltk_data/corpora/assignment/wsj_training/"
dbpedia_path_ttl = "/home/george/PycharmProjects/nlp_assignment/wrd_instances.ttl"
dbpedia_path_csv = "/home/george/PycharmProjects/nlp_assignment/wrd_instances.csv"
dbp_ent_path = "/home/george/PycharmProjects/nlp_assignment/entities.txt"
names = set().union(nltknames.words("male.txt"), nltknames.words("female.txt"))
titles = ["Mr", "Mrs", "Dr", "Sir", "Prof", "Professor", "Ms"]

def train():
    onlyfiles = [f for f in listdir(training_path) if isfile(join(training_path, f))]
    onlyfiles.remove(".DS_Store")

    text = ""
    for f in onlyfiles:
        with open(training_path + f, 'r') as mf:
            text += mf.read()
    print("Files Loaded")

    sentences = nltk.sent_tokenize(text)
    print("Tokenized sentences")

    docs_pattern = r'<ENAMEX TYPE=".*?">.*?</ENAMEX>'
    doc_tuples = re.findall(docs_pattern, text, re.DOTALL)

    entities = []
    i = 0
    for t in doc_tuples:
        i+=1
        #print t
        s = re.compile(r'["<>]')
        t2 = s.split(t)
        tag = t2[2]
        entity = t2[4]
        tagged_entity = nltk.pos_tag(entity.split())

        rtags = map(list, zip(*tagged_entity))
        rtags = rtags[1]

        sys.stdout.write("\r%d%%" % int(i * 100 / len(doc_tuples)))
        sys.stdout.flush()

        entities += [(entity, tagged_entity, rtags, tag)]
        #print entity
        #print tagged_entity
        #print tag

    return entities

def compile_tags(ent_list):
    tag_lists = [" ".join(x[2]) for x in ent_list]

    fq = defaultdict(int)
    for l in tag_lists:
        fq[l] += 1

    return fq


def create_grammar(ent_list):
    illegal_chars = set([":", ")", "("])
    tag_list = [a.split() for a in set([" ".join(x[2]) for x in ent_list])]
    pos_frequencies = compile_tags(ent_list)
    tag_list = sorted(tag_list, key=lambda l: pos_frequencies[" ".join(l)] - len(l)*5)
    grammar = "NE: {" \
              + "}\n{".join(["<" + "><".join(x) + ">"
                                     for x in tag_list
                                     if len(set(x) & illegal_chars) == 0
                                     and ("NNP" in x
                                          or "NNPS" in x
                                          or ("NN" in x
                                              and len(x) == 1))]) \
              + "}"
    return grammar


def get_entities_grammar(grammar):
    onlyfiles = [f for f in listdir(untagged_path) if isfile(join(untagged_path, f))]
    onlyfiles.remove(".DS_Store")

    text = ""
    for f in onlyfiles[:1]:
        with open(untagged_path + f, 'r') as mf:
            text += mf.read()
    print("Files Loaded")

    sentences = nltk.sent_tokenize(text)
    print("Tokenized sentences")
    sentences2 = [nltk.word_tokenize(sent) for sent in sentences]
    print("Tokenized words")
    sentences3 = [nltk.pos_tag(sent) for sent in sentences2]
    print("POS tagged")
    print(sentences3)
    zipsents = [map(list, zip(*sent)) for sent in sentences3]

    if grammar == "": grammar = "NE: {(<NNP|NNPS>+<CC>*)+} # NAMED ENTITY"
    cp = nltk.RegexpParser(grammar)

    entities = []
    i = 0
    for s in sentences3:
        sys.stdout.write("\r%d%%" % int(i * 100 / len(sentences3)))
        sys.stdout.flush()
        i+=1
        t = cp.parse(s)
        for st in t.subtrees():
            if st.label() == "NE":
                entities += [st.leaves()]

    print()
    print("%d entities discovered." % len(entities))

    dbp_ent_set = []
    with open(dbp_ent_path, 'r') as ef:
        dbp_ent_set = ef.read().splitlines()
    dbp_ent_set = set(dbp_ent_set)
    print("%d Entities loaded from entities.txt" % len(dbp_ent_set))
    print("Hercules" in dbp_ent_set)
    print([" ".join([x[0] for x in e]) for e in entities])
    entities = [e for e in entities if " ".join([x[0] for x in e]) in dbp_ent_set]
    print("%d entities remaining following filtering" % len(entities))

    return entities


def extract_entities(e):
    entities = [" ".join(x) for x in [[z[0] for z in y] for y in e]]
    return entities


def intersect(e1, e2):
    return e1 & e2


def run():
    print("")
    print("Collecting test entities")
    print("------------------------")
    test_ents = train()
    test_ents2 = [e[0] for e in test_ents]
    stest_ents2 = set(test_ents2)
    print("%d unique entities collected." % len(stest_ents2))


    print("")
    print("Generating Grammar")
    print("------------------")
    grammar = create_grammar(test_ents)
    print("Grammar created")


    print("")
    print("Testing grammar")
    print("---------------")
    res_ents = extract_entities(get_entities_grammar(grammar))
    sres_ents = set(res_ents)
    print("%d unique entities discovered." % len(sres_ents))


    print("")
    print("Statistics")
    print("----------")
    print("%d unique entities collected from test files." % len(stest_ents2))
    print("%d unique entities discovered from untagged files." % len(sres_ents))
    int_ents = intersect(sres_ents, stest_ents2)
    print("%d common elements (%d%%)." % (len(int_ents), int(len(sres_ents) * 100 / len(stest_ents2))))

    return int_ents, stest_ents2, sres_ents, grammar
