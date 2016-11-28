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
import os
from py_bing_search import PyBingWebSearch

training_path = "/home/george/nltk_data/corpora/assignment/wsj_training/wsj_training/"
untagged_path = "/home/george/nltk_data/corpora/assignment/wsj_untagged/wsj_untagged/"
corpus_root = "/home/george/nltk_data/corpora/assignment/wsj_training/"
dbpedia_path_ttl = "/home/george/PycharmProjects/nlp_assignment/wrd_instances.ttl"
dbpedia_path_csv = "/home/george/PycharmProjects/nlp_assignment/wrd_instances.csv"
dbp_ent_path = "/home/george/PycharmProjects/nlp_assignment/entities.txt"
names = set().union(nltknames.words("male.txt"), nltknames.words("female.txt"))

titles = {"Mr.", "Mrs.", "Dr.", "Sir", "Prof.", "Professor", "Ms.", "Rev.", "President", "Pres.", "Judge", "Mayor"}
business_words = {"Co.", "Company", "Assoc.", "Association", "Inc.", "Incorporated", "Corp.", "Corporation", "Ltd.", "Group", "PLC", "Club", "Court", "Commission", "Bureau", "Ministry", "Institute"}

location_prev_words = {"in"}
person_prev_words = {}
organization_prev_words = {}


def get_ieer_entities():
    entity_dict = {}
    entity_names = []
    for doc in ieer.parsed_docs():
        for st in doc.text.subtrees():
            if st.label() in ["PERSON", "ORGANIZATION", "LOCATION"]:
                entity_names += [" ".join(st.leaves())]
                entity_dict[" ".join(st.leaves())] = st.label()
    return entity_dict, set(entity_names)


def get_training_entities(file_count):
    onlyfiles = [f for f in listdir(training_path) if isfile(join(training_path, f))]
    onlyfiles.remove(".DS_Store")

    onlyfiles = sorted(onlyfiles)

    file_count = min(file_count, len(onlyfiles))
    text = ""
    for f in onlyfiles[:file_count]:
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
        i += 1
        # print t
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
        # print entity
        # print tagged_entity
        # print tag
    print()
    print("%d training entities loaded" % len(entities))
    return entities


def compile_pos_tags(ent_list):
    tag_lists = [(" ".join(x[2]), x[3]) for x in ent_list]

    fq = defaultdict(int)
    for l in tag_lists:
        fq[l] += 1

    return fq


# Create the grammar for the specific type of entity
def create_grammar(ent_list):
    illegal_chars = set([":", ")", "("])

    # Get a list of the POS tags for the entities and their type [(['NNP', 'NNP'], "PERSON")]
    tag_list = [(a[0].split(), a[1]) for a in set([(" ".join(x[2]), x[3]) for x in ent_list])]

    # Sort the list of POS tags so that it's ordered by length to promote greedy matching
    pos_frequencies = compile_pos_tags(ent_list)
    #print pos_frequencies
    avg_freq = sum(pos_frequencies.values()) / len(pos_frequencies)

    tag_list = [x for x in tag_list if pos_frequencies[(" ".join(x[0]), x[1])] > avg_freq * 0.5]
    tag_list = sorted(tag_list, key=lambda l: pos_frequencies[(" ".join(l[0]), l[1])] - (len(l[0]) ** 100))

    # Create the grammar
    grammar_list = [t[1] + ": {<" + "><".join(t[0]) + ">}" for t in tag_list if
                    len(set(t[0]) & illegal_chars) == 0 and (
                    "NNP" in t[0] or "NNPS" in t[0] or ("NN" in t[0] and len(t[0]) == 1))]
    grammar = "\n".join(grammar_list)

    print("Grammar created")
    return grammar


# Delete all files in the untagged directory with extension .result.txt
def delete_files():
    for f in os.listdir(untagged_path):
        if f.endswith(".result"):
            os.remove(untagged_path + f)


def is_name(entity):
    name = entity.split()
    if len(titles & set(name)) > 0:
        return True

    contains_name = False
    for n in name:
        if n in names:
            contains_name = True
        if not n[0].isupper():
            return False
        if "." in n and not len(n) == 2:
            return False
    return contains_name



def is_location(entity):
    if len(entity) <= 7 and (re.match(r'\w+\.', entity)) and entity[0].isupper():
        return True
    return False


def is_organization(entity, past_entities):
    org = entity.split()
    if len(business_words & set(org)) > 0:
        return True
    if entity.isupper() and len(org) == 1 and len(entity) <= 7:
        return True
    past_orgs = [e for e in past_entities if e[1] == "ORGANIZATION"]
    for o in past_orgs:
        org_split = set(o[0].split())
        if len(set(org) & org_split) > 0:
            return True
    return False


def get_relation(entity, ieer_entity_dict, ieer_entity_names, dbp_ent_set, past_entities, prev_word):
    es = entity.split()
    # es = [w for w in es if not w in titles]
    entity_name = " ".join(es)
    entity = "_".join(es)
    entity = re.sub(r'[^\P{P}\w\.\_]+', "", entity)

    if prev_word is not None:
        if prev_word in location_prev_words:
            return "LOCATION"

        if prev_word in person_prev_words:
            return "PERSON"

        if prev_word in organization_prev_words:
            return "ORGANIZATION"

    if entity_name in ieer_entity_names:
        return ieer_entity_dict[entity_name]

    if is_organization(entity_name, past_entities):
        return "ORGANIZATION"

    if is_name(entity_name):
        return "PERSON"

    if is_location(entity_name):
        return "LOCATION"

    if not entity in dbp_ent_set:
        return None

    if len(es) == 1 and entity_name[0].islower():
        return None

    if False:
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
                    return "ORGANIZATION"
                elif v == "http://dbpedia.org/ontology/Location":
                    return "LOCATION"
        except:
            pass

    return None


# Complete the NER
def ner(grammar, file_count):
    dbp_ent_set = None
    with open(dbp_ent_path, 'r') as ef:
        dbp_ent_set = ef.read().splitlines()
    dbp_ent_set = set(dbp_ent_set)
    print("Loaded %d entities from DBPedia entity list" % len(dbp_ent_set))

    delete_files()

    onlyfiles = [f for f in listdir(untagged_path) if isfile(join(untagged_path, f))]
    onlyfiles.remove(".DS_Store")

    onlyfiles = sorted(onlyfiles)

    related_entities = []
    non_related_entities = []

    file_count = min(file_count, len(onlyfiles))

    ieer_dict, ieer_names = get_ieer_entities()

    i = 0
    for f in onlyfiles[:file_count]:
        sys.stdout.write("\r%.2f%%" % (float(i) * 100.0 / float(file_count)))
        sys.stdout.flush()
        i+=1
        text = ""
        with open(untagged_path + f, 'r') as mf:
            text += mf.read()

        # Split into sentences
        sentences_tokenized = nltk.sent_tokenize(text)
        sentences = []
        for sent in sentences_tokenized:
            if "\n" in sent:
                sents = sent.split("\n")
                for s in sents[:-1]:
                    sentences += [s+"\n"]
                if not sents[-1] == "":
                    sentences += [sents[-1]]
            else:
                sentences += [sent]

        tagged_sentences = []
        for sentence in sentences:

            # Tokenize and POS tag the sentence
            sentence2 = nltk.word_tokenize(sentence)
            sentence3 = nltk.pos_tag(sentence2)

            # Parse the sentence using the given grammars
            parser = nltk.RegexpParser(grammar)
            entities = []
            parse_tree = parser.parse(sentence3)

            # Extract the entities from the parse tree
            for subtree in parse_tree.subtrees():
                if subtree.label() in ["PERSON", "ORGANIZATION", "LOCATION"]:
                    entities += [(subtree.leaves(), subtree.label())]

            # Extract the entity names
            named_entities = [(" ".join(x[0]), x[1]) for x in [([z[0] for z in y[0]], y[1]) for y in entities]]

            tagged_sentence = sentence
            for ne in named_entities:

                # Get the word prior to the occurence of the entity in the sentence
                prev_words = sentence.partition(" %s " % ne[0])
                if prev_words[2] == '':
                    prev_word = None
                else:
                    prev_word = prev_words[0].split()[-1]

                # Get an initial relation for the entity
                rel = get_relation(ne[0], ieer_dict, ieer_names, dbp_ent_set, related_entities[-10:], prev_word)
                if rel is not None:
                    related_entities += [(ne[0], rel)]
                    tagged_sentence = tagged_sentence.replace(ne[0], "<ENAMEX TYPE=\"" + rel + "\">" + ne[0] + "</ENAMEX>")
                elif " and " in ne[0]:
                    ne_split = ne[0].split(" and ")
                    for ne_s in ne_split:
                        rel_s = get_relation(ne_s, ieer_dict, ieer_names, dbp_ent_set, related_entities[-10:], None)
                        if rel_s is not None:
                            related_entities += [(ne_s, rel_s)]
                            tagged_sentence = tagged_sentence.replace(ne_s, "<ENAMEX TYPE=\"" + rel_s + "\">" + ne_s + "</ENAMEX>")
                else:
                    non_related_entities += [ne]

            #print("%d, %d" % (len(related_entities), len(non_related_entities)))
            tagged_sentences += [tagged_sentence]
            #print(tagged_sentence)

        #print(" ".join(tagged_sentences))

        with open(untagged_path + f + ".result", 'w') as fi:
            fi.write(" ".join(tagged_sentences))

    print("%d entities successfully extracted and tagged" % len(related_entities))
    print("%d recognised entities rejected" % len(non_related_entities))
    print("%d files searched" % file_count)
    return related_entities, non_related_entities


def statistics(training_entities, related_entities, file_count):
    training_entities = [(x[0], x[3]) for x in training_entities]
    training_entities_set = set(training_entities)
    successes = 0
    failures = 0
    for ent in related_entities:
        if ent in training_entities_set:
            successes += 1
        else:
            failures += 1

    success_percentage = float(successes) * 100.0 / float(len(training_entities))
    print("Using %d files:" % file_count)
    print("%d training entities provided" % len(training_entities))
    print("%d related entities provided" % len(related_entities))
    print("%d successful relations identified" % successes)
    print("%d relations falsely identified" % failures)
    print("%d relations not identified" % (len(training_entities) - successes))
    print("%.2f%% success percentage" % success_percentage)
    return success_percentage


def run(file_count, print_statistics=True):

    print("")
    print("Getting Training Entities")
    print("-------------------------")
    training_entities = get_training_entities(2000)

    print("")
    print("Creating Grammar")
    print("----------------")
    grammar = create_grammar(training_entities)

    print("")
    print("Completing NER")
    print("--------------")
    related_entities, failed_entities = ner(grammar, file_count)

    success_percentage = None
    if print_statistics:
        print("")
        print("Gathering Statistics")
        print("--------------------")
        sub_training_entities = get_training_entities(file_count)
        success_percentage = statistics(sub_training_entities, related_entities, file_count)

    print("")
    print("============")
    print("# COMPLETE #")
    print("============")

    missing_relations = set([(x[0], x[3]) for x in training_entities]) - (set(related_entities) | set(failed_entities))

    return training_entities, grammar, related_entities, failed_entities, missing_relations, success_percentage

    # remove all .result.txt files
    # for file in files:
    #    for each sentence:
    #        POS tag
    #        identify entities
    #        relate entity
    #        replace entity in sentence
    #        add sentence to text
    #    write text to file with .result.txt extension
