from os import listdir
from os.path import isfile, join
import nltk
from nltk.corpus import ieer
from nltk.corpus import names as nltknames
from SPARQLWrapper import SPARQLWrapper, JSON
import re
from nltk.tokenize.punkt import PunktSentenceTokenizer
import sys
from collections import defaultdict
import time
import os

training_path = "/home/george/nltk_data/corpora/assignment/nlp_training/training/"
untagged_path = "/home/george/nltk_data/corpora/assignment/nlp_untagged/"
test_path_tagged = "/home/george/nltk_data/corpora/assignment/test_tagged/"
test_path_untagged = "/home/george/nltk_data/corpora/assignment/test_untagged/"

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
    return names[0]


def parse_time(time_text):
    time_parse_regex = re.compile(r'(\d{1,2})([:.,](\d{2}))?(\s?([aApP]\.?[mM]\.?))?')
    hrs, _, mns, _, pm = re.findall(time_parse_regex, time_text)[0]
    if not pm == '':
        if (pm[0] == 'p' or pm[0] == 'P') and int(hrs) < 12:
            hrs = str((int(hrs) + 12)) + ""
    return hrs + mns


def train_sent_tokenizer():
    onlyfiles = [f for f in listdir(training_path) if isfile(join(training_path, f))]
    if ".DS_Store" in onlyfiles:
        onlyfiles.remove(".DS_Store")

    sentences = []
    for f in onlyfiles[:20]:
        with open(training_path + f, 'r') as mf:
            text = mf.read()
        #print(text)
        sents_pattern = r'<sentence>.*?</sentence>'
        sents = re.findall(sents_pattern, text, re.DOTALL)
        #print(sents)
        sentences += [x[10:-11] for x in sents]

    #print sentences
    tokenizer = PunktSentenceTokenizer()
    tokenizer.train(". ".join(sentences))
    return tokenizer


def process_abstract(abstract, header_dict):

    time_regex = re.compile(r'(\d{1,2}([:.,]\d{2})?(\s?([aApP]\.?[mM]\.?))?)')
    times = re.findall(time_regex, abstract)
    times = [x[0] for x in times]

    if 'stime' in header_dict:
        for t in times:
            if parse_time(t) == header_dict['stime']:
                abstract = abstract.replace(t, "<stime>" + t + "</stime>")

    if 'etime' in header_dict:
        for t in times:
            if parse_time(t) == header_dict['etime']:
                abstract = abstract.replace(t, "<etime>" + t + "</etime>")

    if 'location' in header_dict:
        abstract = abstract.replace(header_dict['location'], "<location>" + header_dict['location'] + "</location>")

    if 'speaker' in header_dict:
        abstract = abstract.replace(header_dict['speaker'], "<speaker>" + header_dict['speaker'] + "</speaker>")

    # Split up the sentences in the abstract
    tokenizer = train_sent_tokenizer()
    sentences_tokenized = tokenizer.tokenize(abstract)
    print(sentences_tokenized)
    text = ""

    speaker_set = set()

    # Add tags to the sentences
    for sentence in sentences_tokenized:

        # Process sentence tags for the sentence
        if sentence[0:1] == "\n":
            new_sent = "\n<sentence>" + sentence[1:-1] + "</sentence>."
        else:
            new_sent = "<sentence>" + sentence[:-1] + "</sentence>."

        # Parse the sentence using the given grammars
        pos_tagged_sent = nltk.pos_tag(nltk.word_tokenize(sentence))
        parser = nltk.RegexpParser("SP: {<NNP><NNP>}")
        entities = []
        parse_tree = parser.parse(pos_tagged_sent)
        #print(parse_tree)

        for subtree in parse_tree.subtrees():
            if subtree.label() == "SP":
                entities += [subtree.leaves()]
        speakers = [" ".join([y[0] for y in z]) for z in entities]
        print(speakers)

        for speaker in speakers:
            if not ('Host' in header_dict and header_dict['Host'] != speaker):
                if speaker in speaker_set or is_name(speaker):
                    speaker_set.add(speaker)
                    new_sent = new_sent.replace(speaker, "<speaker>" + speaker + "</speaker>")

        text += new_sent

    return text


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
                    header_dict["stime"] = parse_time(st)
                if et is not None:
                    new_line = new_line.replace(et, "<etime>" + et + "</etime>")
                    header_dict["etime"] = parse_time(et)

            # Check if the location can be matched
            if t == "Place":
                new_line = new_line.replace(v.lstrip(), "<location>" + v.lstrip() + "</location>")
                header_dict['location'] = v.lstrip()

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
    #print(result_text)
    print(abstract)
    result_text += abstract
    return result_text


def get_file_contents(file):
    with open(file, 'r') as f:
        contents = f.read()
    return contents
