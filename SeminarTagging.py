from os import listdir
from os.path import isfile, join
import nltk
from nltk.corpus import ieer
from nltk.corpus import names as nltknames
from SPARQLWrapper import SPARQLWrapper, JSON
import re
import sys
from collections import defaultdict
import time
import os

training_path = "/home/george/nltk_data/corpora/assignment/nlp_training/training/"
untagged_path = "/home/george/nltk_data/corpora/assignment/nlp_untagged/"
test_path = "/home/george/nltk_data/corpora/assignment/wsj_New_test_data/"

dbpedia_path_ttl = "/home/george/PycharmProjects/nlp_assignment/wrd_instances.ttl"
dbpedia_path_csv = "/home/george/PycharmProjects/nlp_assignment/wrd_instances.csv"
dbp_ent_path = "/home/george/PycharmProjects/nlp_assignment/entities.txt"
more_entities = "/home/george/PycharmProjects/nlp_assignment/more_entities.txt"
names = set().union(nltknames.words("male.txt"), nltknames.words("female.txt"))
titles = {"Mr.", "Mrs.", "Dr.", "Sir", "Prof.", "Professor", "Ms.", "Rev.", "President", "Pres.", "Judge", "Mayor",
          "Sr", "Jr"}


# \d{1,2}([:.,]\d{2})?(\s?([aApP]\.?[mM]\.?))?

# All files start with a line enclosed by "<>"
# followed by several lines of "tag:" and their contents
# ending in "Abstract:"


def get_times(text):
    time_regex = re.compile(r'(\d{1,2}([:.,]\d{2})?(\s?([aApP]\.?[mM]\.?))?)')
    times = re.findall(time_regex, text)
    if len(times) == 1:
        return times[0][0], None
    if len(times) > 1:
        return times[0][0], times[1][0]
    else:
        return None, None


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


def extract_name(text):
    names = text.split(", ")
    for n in names:
        if is_name(n):
            return n
    return None


def parse_date(time_text):
    time_parse_regex = re.compile(r'(\d{1,2})([:.,](\d{2}))?(\s?([aApP]\.?[mM]\.?))?')
    hrs, _, mns, _, pm = re.findall(time_parse_regex, time_text)[0]
    if not pm == '':
        if (pm[0] == 'p' or pm[0] == 'P') and int(hrs) < 12:
            hrs = str((int(hrs) + 12)) + ""
    return hrs + mns


def process_abstract(abstract, header_dict):
    sentences_tokenized = nltk.sent_tokenize(abstract)
    sentences = []
    for sent in sentences_tokenized:
        if "\n" in sent:
            sents = sent.split("\n")
            for s in sents[:-1]:
                sentences += [s + "\n"]
            if not sents[-1] == "":
                sentences += [sents[-1]]
        else:
            sentences += [sent]

    tagged_sentences = []
    for sentence in sentences:

        # Tokenize and POS tag the sentence
        sentence2 = nltk.word_tokenize(sentence)
        sentence3 = nltk.pos_tag(sentence2)
        print sentence3
    """
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
                spl_prev_word = prev_words[0].split()
                if spl_prev_word != []:
                    prev_word = spl_prev_word[-1]
                else:
                    prev_word = None

            # Get an initial relation for the entity
            rel = get_relation(ne[0], ieer_dict, ieer_names, dbp_ent_set, sample_entities, related_entities[-10:],
                               prev_word)
            if rel is not None:
                related_entities += [(ne[0], rel)]
                tagged_sentence = tagged_sentence.replace(ne[0],
                                                          "<ENAMEX TYPE=\"" + rel + "\">" + ne[0] + "</ENAMEX>")
            elif " and " in ne[0]:
                ne_split = ne[0].split(" and ")
                for ne_s in ne_split:
                    rel_s = get_relation(ne_s, ieer_dict, ieer_names, dbp_ent_set, sample_entities,
                                         related_entities[-10:], None)
                    if rel_s is not None:
                        related_entities += [(ne_s, rel_s)]
                        tagged_sentence = tagged_sentence.replace(ne_s,
                                                                  "<ENAMEX TYPE=\"" + rel_s + "\">" + ne_s + "</ENAMEX>")
            else:
                non_related_entities += [ne]

        # print("%d, %d" % (len(related_entities), len(non_related_entities)))
        tagged_sentences += [tagged_sentence]
        # print(tagged_sentence)
    """
    return None


def process_training_file(file_contents):
    # Break the file by lines
    content_split = file_contents.split("\n")

    # Check that the first line starts and ends with "<>"
    if content_split[0][0] == "<" and content_split[0][-1] == ">":
        header_line = content_split[0]
    else:
        header_line = None

    # Get a list of the lines up to "Abstract:"
    result_text = header_line
    header_dict = {}
    header_regex = re.compile(r'\w+\:.')
    i = 1
    # Put all of the lines into a dictionary until we get to abstract
    while not content_split[i].startswith("Abstract:"):

        # Check that the line matches the specified type
        if re.match(header_regex, content_split[i]):

            new_line = content_split[i]

            # Add the line to the dictionary
            t, _, v = content_split[i].partition(":")
            prev_t = t
            header_dict[t] = v.lstrip()

            leading_spaces = len(v) - len(v.lstrip(' '))

            # Check if the times can be matched
            if t == "Time":
                st, et = get_times(v)
                if st is not None:
                    new_line = new_line.replace(st, "<stime>" + st + "</stime>")
                    header_dict["start_time"] = parse_date(st)
                if et is not None:
                    new_line = new_line.replace(et, "<etime>" + et + "</etime>")
                    header_dict["end_time"] = parse_date(et)

            # Check if the location can be matched
            if t == "Place":
                new_line = new_line.replace(v.lstrip(), "<location>" + v.lstrip() + "</location>")

            # Check if the speaker can be matched
            if t == "Host":
                name = extract_name(v.lstrip())
                new_line = new_line.replace(name, "<speaker>" + name + "</speaker>")
                header_dict["speaker"] = name


            result_text += "\n" + new_line

        # If it isn't then it must belong to the previous line
        else:

            # Attach it to the item that was added previously
            header_dict[prev_t] += " " + content_split[i].lstrip()

        i += 1

    # The remainder of the file is the abstract
    abstract = content_split[(i + 1):]
    abstract = "\n".join(abstract)
    abstract = process_abstract(abstract, header_dict)
    print(result_text)
    print(abstract)
    return result_text


def get_file_contents(file):
    with open(file, 'r') as f:
        contents = f.read()
    return contents
