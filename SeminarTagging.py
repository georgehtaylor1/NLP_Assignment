from os import listdir
from os.path import isfile, join
import nltk
from nltk.corpus import names as nltknames
import re
from nltk.tokenize.punkt import PunktSentenceTokenizer
import os
from collections import defaultdict

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

# All files start with a line enclosed by "<>"
# followed by several lines of "tag:" and their contents
# ending in "Abstract:"


# Extract any times from the text
def get_times(text):
    time_regex = re.compile(r'(\d{1,2}([:.,]\d{2})?(\s?([aApP]\.?[mM]\.?))?)')
    times = re.findall(time_regex, text)
    if len(times) == 1:
        return times[0][0], None
    if len(times) > 1:
        return times[0][0], times[1][0]
    else:
        return None, None


# Determine if the entity is a name
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


# Extract a name from the header field
def extract_name(text):
    names = text.split(", ")
    return names[0]


# Parse the time into a 24 hour format HHMM
def parse_time(time_text):
    time_parse_regex = re.compile(r'(\d{1,2})([:.,](\d{2}))?(\s?([aApP]\.?[mM]\.?))?')
    hrs, _, mns, _, pm = re.findall(time_parse_regex, time_text)[0]
    if not pm == '':
        if (pm[0] == 'p' or pm[0] == 'P') and int(hrs) < 12:
            hrs = str((int(hrs) + 12)) + ""
    return hrs + mns


# Train the sentence tokenizer on the training data
def train_sent_tokenizer():
    onlyfiles = [f for f in listdir(training_path) if isfile(join(training_path, f))]
    if ".DS_Store" in onlyfiles:
        onlyfiles.remove(".DS_Store")

    sentences = []
    for f in onlyfiles[:20]:
        with open(training_path + f, 'r') as mf:
            text = mf.read()

        sents_pattern = r'<sentence>.*?</sentence>'
        sents = re.findall(sents_pattern, text, re.DOTALL)
        sentences += [x[10:-11] for x in sents]

    tokenizer = PunktSentenceTokenizer()
    tokenizer.train(". ".join(sentences))
    return tokenizer


# Process the abstract of the seminar
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

    if 'who' in header_dict:
        abstract = abstract.replace(header_dict['who'], "<speaker>" + header_dict['who'] + "</speaker>")

    paragraphs = abstract.split("\n\n")
    text = ""

    for paragraph in paragraphs:
        # Split up the sentences in the abstract
        tokenizer = train_sent_tokenizer()
        sentences_tokenized = tokenizer.tokenize(paragraph)
        paragraph_text = ""

        speaker_set = set()

        # Add tags to the sentences
        for sentence in sentences_tokenized:

            new_sent = "<sentence>" + sentence + "</sentence>"

            if "who" not in header_dict and "speaker" not in header_dict:
                # Parse the sentence using the given grammar
                pos_tagged_sent = nltk.pos_tag(nltk.word_tokenize(sentence))
                parser = nltk.RegexpParser("SP: {<NNP><NNP><NNP>}\nSP: {<NNP><NNP>}")
                entities = []
                parse_tree = parser.parse(pos_tagged_sent)

                for subtree in parse_tree.subtrees():
                    if subtree.label() == "SP":
                        entities += [subtree.leaves()]
                speakers = [" ".join([y[0] for y in z]) for z in entities]

                for speaker in speakers:
                    if not ('Host' in header_dict and header_dict['Host'] != speaker):
                        if speaker in speaker_set or is_name(speaker):
                            speaker_set.add(speaker)
                            new_sent = new_sent.replace(speaker, "<speaker>" + speaker + "</speaker>")

            paragraph_text += new_sent

        text += "<paragraph>" + paragraph_text + "</paragraph>\n\n"

    return text


# Process the contents of the file (header + abstract)
def process_file_contents(file_contents):
    # Break the file by lines
    content_split = file_contents.split("\n")

    # Check that the first line starts and ends with "<>"
    if content_split[0][0] == "<" and content_split[0][-1] == ">":
        header_line = content_split[0]
    else:
        header_line = None

    if header_line is not None:
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
                header_dict[t.lower()] = v.lstrip()

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

                if t == "Who":
                    header_dict['who'] = extract_name(v.lstrip())

                result_text += "\n" + new_line

            # If it isn't then it must belong to the previous line
            else:

                # Attach it to the item that was added previously
                header_dict[prev_t.lower()] += " " + content_split[i].lstrip()

            i += 1

        # The remainder of the file is the abstract
        abstract = content_split[(i + 1):]
        abstract = "\n".join(abstract)
        abstract = "\nAbstract:\n\n" + process_abstract(abstract, header_dict)

        result_text += abstract
        result_text = result_text.replace("<paragraph></paragraph>", "")
        return result_text, header_dict
    else:
        return None, None


# Get the contents of the specified file
def get_file_contents(file):
    with open(file, 'r') as f:
        contents = f.read()
    return contents


# Delete all files in the untagged directory with extension .result.txt
def delete_files(active_path):
    for f in os.listdir(active_path):
        if f.endswith(".result"):
            os.remove(active_path + f)


# Process the given file from start to finish
def process_file(file):
    content = get_file_contents(file)
    result = process_file_contents(content)
    with open(file + ".result", 'w') as f:
        f.write(result[0])
    return result


# Load the entities from the given file
def load_entities(file):
    contents = get_file_contents(file)
    ent_dict = defaultdict()

    stime_pattern = re.compile(r'<stime>.*?</stime>')
    stimes = re.findall(stime_pattern, contents)
    if len(stimes) > 0:
        ent_dict['stime'] = stimes[0]

    etime_pattern = re.compile(r'<etime>.*?</etime>')
    etimes = re.findall(etime_pattern, contents)
    if len(etimes) > 0:
        ent_dict['etime'] = etimes[0]

    speaker_pattern = re.compile(r'<speaker>.*?</speaker>')
    speakers = re.findall(speaker_pattern, contents)
    if len(speakers) > 0:
        ent_dict['speakers'] = speakers[0]

    location_pattern = re.compile(r'<location>.*?</location>')
    locations = re.findall(location_pattern, contents)
    if len(locations) > 0:
        ent_dict['location'] = locations[0]

    return ent_dict


# Check if a tag is present in both dictionaries and if it is the same
def check_tag(tag, tagged_dict, untagged_dict):
    if tag in tagged_dict and tag in untagged_dict:
        return tagged_dict[tag] == untagged_dict[tag]
    elif tag in tagged_dict or tag in untagged_dict:
        return False
    else:
        return True


# Test the accuracy of the tagged files
def test():
    untagged_files = [f for f in listdir(test_path_untagged) if isfile(join(test_path_untagged, f)) and f.endswith("result")]
    if ".DS_Store" in untagged_files:
        untagged_files.remove(".DS_Store")
    untagged_files = sorted(untagged_files)

    tagged_files = [f for f in listdir(test_path_tagged) if isfile(join(test_path_tagged, f))]
    if ".DS_Store" in tagged_files:
        tagged_files.remove(".DS_Store")
    tagged_files = sorted(tagged_files)

    successes = 0
    failures = 0

    for tagged, untagged in zip(tagged_files, untagged_files):
        tagged_dict = load_entities(test_path_tagged + tagged)
        untagged_dict = load_entities(test_path_untagged + untagged)

        tags = ['stime', 'etime', 'location', 'speaker']
        bools = map(lambda x: check_tag(x, tagged_dict, untagged_dict), tags)

        if all(bools):
            successes += 1
        else:
            failures += 1

    return successes, failures


# Run the seminar tagging
def run():
    delete_files(test_path_untagged)
    onlyfiles = [f for f in listdir(test_path_untagged) if isfile(join(test_path_untagged, f))]
    if ".DS_Store" in onlyfiles:
        onlyfiles.remove(".DS_Store")

    onlyfiles = sorted(onlyfiles)

    results = []

    for f in onlyfiles:
        print("Working on %s" % f)
        results += [process_file(test_path_untagged + f)]

    return results
