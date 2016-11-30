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



# All files start with a line enclosed by "<>"
# followed by several lines of "tag:" and their contents
# ending in "Abstract:"
def process_training_file(file_contents):

    # Break the file by lines
    content_split = file_contents.split("\n")

    # Check that the first line starts and ends with "<>"
    if content_split[0][0] == "<" and content_split[0][-1] == ">":
        header_line = content_split[0]
    else:
        header_line = None

    # Get a list of the lines up to "Abstract:"
    header_dict = {}
    header_regex = re.compile(r'\w+\:.')
    i = 1
    # Put all of the lines into a dictionary until we get to abstract
    while not content_split[i].startswith("Abstract:"):

        # Check that the line matches the specified type
        if re.match(header_regex, content_split[i]):

            # Add the line to the dictionary
            t, _, v = content_split[i].partition(":")
            prev_t = t
            header_dict[t] = v.lstrip()

        # If it isn't then it must belong to the previous line
        else:

            # Attach it to the item that was added previously
            header_dict[prev_t] += " " + content_split[i].lstrip()

        i+=1

    # The remainder of the file is the abstract
    abstract = content_split[(i+1):]

    print(header_line)
    print(header_dict)
    print("\n".join(abstract))
    return None


def get_file_contents(file):
    with open(file, 'r') as f:
        contents = f.read()
    return contents
