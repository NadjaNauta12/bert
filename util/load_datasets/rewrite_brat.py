import codecs
import os
from enum import Enum
import util.load_datasets.brat_annotations as brat_annotations
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import re
from itertools import groupby
import difflib
from collections import Counter

'''Simple class for representing the desired output'''
class CoNLL_Token:
    def __init__(self, token, start, end, token_label=None, sentence_label=None, is_end_of_sentence=False, file=None):
        self.token = token
        self.start = start
        self.end = end
        if token_label is not None:
            self.token_label = token_label
        else:
            self.token_label = Token_Label.OUTSIDE
        self.sentence_label = sentence_label
        self.matched = False
        self.is_end_of_sentence = is_end_of_sentence
        self.file = file

'''Simple class for representing documents'''
class Document:
    def __init__(self, text, file):
        self.text = text
        self.file = file
        self.sentences = [token for token in span_tokenize_sentences(text)]
        self.tokens = [token for token in span_tokenize(text)]

'''Enum for representing our argument labels'''
class Token_Label(Enum):
    BEGIN_BACKGROUND_CLAIM = 1
    INSIDE_BACKGROUND_CLAIM = 2
    BEGIN_OWN_CLAIM = 3
    INSIDE_OWN_CLAIM = 4
    BEGIN_DATA = 5
    INSIDE_DATA = 6
    OUTSIDE = 7


'''Loads and parses the annotation files created with BRAT'''
def load_annotations(path):
    return brat_annotations.parse_annotations(path)


# TODO: Filter out the xml header?
'''Reads the text documents'''
def read_texts(path):
    texts = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if '.txt' in file:
                with codecs.open(os.path.join(subdir, file), mode='r', encoding='utf-8') as f:
                    doc = Document(text=f.read(), file=file)
                    texts.append(doc)
                f.close()
    return texts


'''Takes the BRAT annotations and the corresponding text files and joins them accordingly'''
def join_argument_annotations_and_texts(texts, annotations):
    for doc in texts:
        for ann in annotations:
            if doc.essay.split('.txt')[0] == ann.file.split('.ann')[0]:
                if ann.type == Type.ENTITY:
                    for token in doc.tokens:
                        try:
                            # TODO: potential problem here: The token start might not be the ann start even for the first token, because there might be a problem with the annotation?
                            # TODO: Fix this problem
                            # TODO: Check whether everything worked fine with parts of the same
                            if token.start == ann.start and ann.label == brat_annotations.Label.OWN_CLAIM:
                                token.token_label = Token_Label.BEGIN_OWN_CLAIM
                            elif token.start >= ann.start and token.end <= ann.end and ann.label == brat_annotations.Label.OWN_CLAIM:
                                token.token_label = Token_Label.INSIDE_OWN_CLAIM
                            elif token.start == ann.start and ann.label == brat_annotations.Label.BACKGROUND_CLAIM:
                                token.token_label = Token_Label.BEGIN_BACKGROUND_CLAIM
                            elif token.start >= ann.start and token.end <= ann.end and ann.label == brat_annotations.Label.BACKGROUND_CLAIM:
                                token.token_label = Token_Label.INSIDE_BACKGROUND_CLAIM
                            elif token.start == ann.start and ann.label == brat_annotations.Label.DATA:
                                token.token_label = Token_Label.BEGIN_DATA
                            elif token.start >= ann.start and token.end <= ann.end and ann.label == brat_annotations.Label.DATA:
                                token.token_label = Token_Label.INSIDE_DATA
                        except Exception as e:
                            print(e)


'''Takes the BRAT annotations and the corresponding text files and joins them accordingly'''
def join_dri_and_arg_annotations(discourse_annotations, argument_annotations):
    def keyfunc(val):
        return val.file

    argument_groups = []

    for k, g in groupby(argument_annotations, keyfunc):
        argument_groups.append(list(g))

    for i,discourse_ann in enumerate(discourse_annotations):
        tokens = [w for w in word_tokenize(discourse_ann.text)]
        offset = 0
        discourse_ann.tokens = []
        for j,token in enumerate(tokens):
            position = discourse_ann.text.find(token, offset)
            if position < 0:
                if token == "\'\'" or token == "``":
                    token = '"'
                    position = discourse_ann.text.find(token, offset)
            if position < 0:
                print("Problem")
            else:
                start = position
                end = position + len(token)
                offset = end
                tok = CoNLL_Token(token=token, sentence_label=discourse_ann.category, start=discourse_ann.start + start, end=discourse_ann.start + end, file=discourse_ann.file)
                if j == len(tokens)-1:
                    tok.is_end_of_sentence = True
                discourse_annotations[i].tokens.append(tok)
        #discourse_ann.tokens = [CoNLL_Token(token=w, sentence_label=discourse_ann.category, start=) for w in word_tokenize(discourse_ann.text)]
        try:
            for token in discourse_ann.tokens:
                for arguments in argument_groups:
                    if arguments[0].file.split('.ann')[0] == discourse_ann.file.split('_')[0]:
                        for arg in arguments:
                            if arg.type == brat_annotations.Type.ENTITY:
                                if token.start == arg.start and arg.label == brat_annotations.Label.OWN_CLAIM:
                                    token.token_label = Token_Label.BEGIN_OWN_CLAIM
                                    break
                                elif token.start >= arg.start and token.end <= arg.end and arg.label == brat_annotations.Label.OWN_CLAIM:
                                    token.token_label = Token_Label.INSIDE_OWN_CLAIM
                                    break
                                elif token.start == arg.start and arg.label == brat_annotations.Label.BACKGROUND_CLAIM:
                                    token.token_label = Token_Label.BEGIN_BACKGROUND_CLAIM
                                    break
                                elif token.start >= arg.start and token.end <= arg.end and arg.label == brat_annotations.Label.BACKGROUND_CLAIM:
                                    token.token_label = Token_Label.INSIDE_BACKGROUND_CLAIM
                                    break
                                elif token.start == arg.start and arg.label == brat_annotations.Label.DATA:
                                    token.token_label = Token_Label.BEGIN_DATA
                                    break
                                elif token.start >= arg.start and token.end <= arg.end and arg.label == brat_annotations.Label.DATA:
                                    token.token_label = Token_Label.INSIDE_DATA
                                    break
        except Exception as e:
            print(e)
    return discourse_annotations


def assign_sentence_pos(sentence_list):
    for i,sentence in enumerate(sentence_list):
        for token in sentence.tokens:
            token.sentence_position = i

def join_multiple(annotations_list):
    for list in annotations_list:
        assign_sentence_pos(list)

    ann_list1 = [token for sentence in annotations_list[0] for token in sentence.tokens]
    ann_list2 = [token for sentence in annotations_list[1] for token in sentence.tokens]
    ann_list3 = [token for sentence in annotations_list[2] for token in sentence.tokens]

    #ann_list2_copy = ann_list2[:]
    #ann_list3_copy = ann_list3[:]

    joined_data = []

    def keyfunc_sorting(val):
        return (val.file.split("_")[0], val.start)

    ann_list1 = sorted(ann_list1, key=keyfunc_sorting, reverse=False)
    ann_list2 = sorted(ann_list2, key=keyfunc_sorting, reverse=False)
    ann_list3 = sorted(ann_list3, key=keyfunc_sorting, reverse=False)

    for i,ann1 in enumerate(ann_list1):
        ann1.joined_labels = {"aspect": ann1.sentence_label, "discourse": "DRI_Unspecified", "summary": "NONE"}
        for j,ann2 in enumerate(ann_list2):
            if ann1.file.split('_')[0] == ann2.file.split('_')[0] and ann1.start == ann2.start and ann1.end == ann2.end and ann1.token_label == ann2.token_label:
                ann1.joined_labels["discourse"] = ann2.sentence_label
                del ann_list2[j]
                break
        for k, ann3 in enumerate(ann_list3):
             if ann1.file.split('_')[0] == ann3.file.split('_')[0] and ann1.start == ann3.start and ann1.end == ann3.end and ann1.token_label == ann3.token_label:
                if hasattr(ann1, 'joined_labels'):
                    ann1.joined_labels["summary"] = ann3.sentence_label
                else:
                    print("This should not happen.")
                try:
                    del ann_list3[k]
                except Exception as e:
                    print(e)
                break
        joined_data.append(ann1)

    #ann_list3_copy_copy = ann_list3_copy[:]

    print("The first lists are joined.")
    for i,ann2 in enumerate(ann_list2):
        ann2.joined_labels = {"discourse": ann2.sentence_label, "aspect": "NONE", "summary": "NONE"}
        for j,ann3 in enumerate(ann_list3):
            if ann2.file.split('_')[0] == ann3.file.split('_')[0] and ann2.start == ann3.start and ann2.end == ann3.end and ann2.token_label == ann3.token_label:
                ann2.joined_labels["summary"] = ann3.sentence_label
                del ann_list3[j]
                break
        joined_data.append(ann2)

    for ann3 in ann_list3:
        ann3.joined_labels = {"discourse": "DRI_Unspecified", "aspect": "NONE", "summary": ann3.sentence_label}
        joined_data.append(ann3)

    print("Done")
    occurrences_aspect = Counter([token.joined_labels["aspect"] for token in joined_data])
    occurrences_discourse = Counter([token.joined_labels["discourse"] for token in joined_data])
    occurrences_summary = Counter([token.joined_labels["summary"] for token in joined_data])
    occurrences_arg = Counter([token.token_label for token in joined_data])
    print("Stats aspect: ", occurrences_aspect)
    print("Stats discourse: ", occurrences_discourse)
    print("Stats summary: ", occurrences_summary)
    print("Stats arg: ", occurrences_arg)
    return joined_data


def join_all(annotations_list):
    for list in annotations_list:
        assign_sentence_pos(list)

    ann_list1 = [token for sentence in annotations_list[0] for token in sentence.tokens]
    ann_list2 = [token for sentence in annotations_list[1] for token in sentence.tokens]
    ann_list3 = [token for sentence in annotations_list[2] for token in sentence.tokens]
    ann_list4 = [token for sentence in annotations_list[3] for token in sentence.tokens]


    joined_data = []

    def keyfunc_sorting(val):
        return (val.file.split("_")[0], val.start)

    ann_list1 = sorted(ann_list1, key=keyfunc_sorting, reverse=False)
    ann_list2 = sorted(ann_list2, key=keyfunc_sorting, reverse=False)
    ann_list3 = sorted(ann_list3, key=keyfunc_sorting, reverse=False)
    ann_list4 = sorted(ann_list4, key=keyfunc_sorting, reverse=False)

    for i, ann1 in enumerate(ann_list1):
        ann1.joined_labels = {"aspect": ann1.sentence_label, "discourse": "DRI_Unspecified", "summary": "NONE", "citation": "NONE"}
        for j, ann2 in enumerate(ann_list2):
            if ann1.file.split('_')[0] == ann2.file.split('_')[0] and ann1.start == ann2.start and ann1.end == ann2.end and ann1.token_label == ann2.token_label:
                ann1.joined_labels["discourse"] = ann2.sentence_label
                del ann_list2[j]
                break
        for k, ann3 in enumerate(ann_list3):
            if ann1.file.split('_')[0] == ann3.file.split('_')[0] and ann1.start == ann3.start and ann1.end == ann3.end and ann1.token_label == ann3.token_label:
                if hasattr(ann1, 'joined_labels'):
                    ann1.joined_labels["summary"] = ann3.sentence_label
                else:
                    print("This should not happen.")
                try:
                    del ann_list3[k]
                except Exception as e:
                    print(e)
                break
        for l, ann4 in enumerate(ann_list4):
            if ann1.file.split('_')[0] == ann4.file.split('_')[0] and ann1.start == ann4.start and ann1.end == ann4.end and ann1.token_label == ann4.token_label:
                if hasattr(ann1, 'joined_labels'):
                    ann1.joined_labels["citation"] = ann4.sentence_label
                else:
                    print("This should not happen.")
                try:
                    del ann_list4[l]
                except Exception as e:
                    print(e)
                break
        joined_data.append(ann1)

    for i, ann2 in enumerate(ann_list2):
        ann2.joined_labels = {"discourse": ann2.sentence_label, "aspect": "NONE", "summary": "NONE", "citation": "NONE"}
        for j, ann3 in enumerate(ann_list3):
            if ann2.file.split('_')[0] == ann3.file.split('_')[0] and ann2.start == ann3.start and ann2.end == ann3.end and ann2.token_label == ann3.token_label:
                ann2.joined_labels["summary"] = ann3.sentence_label
                del ann_list3[j]
                break
        for k, ann4 in enumerate(ann_list4):
            if ann2.file.split('_')[0] == ann4.file.split('_')[0] and ann2.start == ann4.start and ann2.end == ann4.end and ann2.token_label == ann4.token_label:
                ann2.joined_labels["citation"] = ann4.sentence_label
                del ann_list4[k]
                break
        joined_data.append(ann2)


    for i, ann3 in enumerate(ann_list3):
        ann3.joined_labels = {"discourse": "DRI_Unspecified", "aspect": "NONE", "summary": ann3.sentence_label, "citation": "NONE"}
        for j, ann3 in enumerate(ann_list3):
            if ann3.file.split('_')[0] == ann4.file.split('_')[0] and ann3.start == ann4.start and ann3.end == ann4.end and ann3.token_label == ann4.token_label:
                ann3.joined_labels["citation"] = ann4.sentence_label
                del ann_list4[j]
                break
        joined_data.append(ann3)

    for ann4 in ann_list4:
        ann4.joined_labels = {"discourse": "DRI_Unspecified", "aspect": "NONE", "summary": "NONE", "citation": ann4.sentence_label}
        joined_data.append(ann4)

    print("Done")
    occurrences_aspect = Counter([token.joined_labels["aspect"] for token in joined_data])
    occurrences_discourse = Counter([token.joined_labels["discourse"] for token in joined_data])
    occurrences_summary = Counter([token.joined_labels["summary"] for token in joined_data])
    occurrences_citation = Counter([token.joined_labels["citation"] for token in joined_data])
    occurrences_arg = Counter([token.token_label for token in joined_data])
    print("Stats aspect: ", occurrences_aspect)
    print("Stats discourse: ", occurrences_discourse)
    print("Stats summary: ", occurrences_summary)
    print("Stats citation: ", occurrences_citation)
    print("Stats arg: ", occurrences_arg)
    return joined_data





        # if doc.file.split('.txt')[0] == ann.file.split('.ann')[0]:
            #     if ann.type == brat_annotations.Type.ENTITY:
            #         for token in doc.tokens:
            #             try:
            #                 # TODO: potential problem here: The token start might not be the ann start even for the first token, because there might be a problem with the annotation?
            #                 # TODO: Fix this problem
            #                 # TODO: Check whether everything worked fine with parts of the same
            #                 if token.start == ann.start and ann.label == brat_annotations.Label.OWN_CLAIM:
            #                     token.token_label = Token_Label.BEGIN_OWN_CLAIM
            #                 elif token.start >= ann.start and token.end <= ann.end and ann.label == brat_annotations.Label.OWN_CLAIM:
            #                     token.token_label = Token_Label.INSIDE_OWN_CLAIM
            #                 elif token.start == ann.start and ann.label == brat_annotations.Label.BACKGROUND_CLAIM:
            #                     token.token_label = Token_Label.BEGIN_BACKGROUND_CLAIM
            #                 elif token.start >= ann.start and token.end <= ann.end and ann.label == brat_annotations.Label.BACKGROUND_CLAIM:
            #                     token.token_label = Token_Label.INSIDE_BACKGROUND_CLAIM
            #                 elif token.start == ann.start and ann.label == brat_annotations.Label.DATA:
            #                     token.token_label = Token_Label.BEGIN_DATA
            #                 elif token.start >= ann.start and token.end <= ann.end and ann.label == brat_annotations.Label.DATA:
            #                     token.token_label = Token_Label.INSIDE_DATA
            #             except Exception as e:
            #                 print(e)


def get_char_positions_for_dri_annotations(annotations, path):
    # group annotations by file first in order to increase matching performance

    def keyfunc(val):
        return val.file

    groups = []

    for k, g in groupby(annotations, keyfunc):
        groups.append(list(g))

    for annotations in groups:
        try:
            with codecs.open(os.path.join(path, annotations[0].file.split('_')[0] + '.txt'), mode='r', encoding='utf-8') as f:
                text = f.read()
                offset = 0
                for i, ann in enumerate(annotations):
                    position = text.find(ann.text, offset)
                    if position >= 0:
                        offset = position + len(ann.text)
                        ann.start = position
                        ann.end = offset
                        ann.text = text[ann.start:ann.end]
                    else:
                        # TODO: The only problem that is not solved 100%: If the ann.text itself contains \n, my counting is of course wrong
                        d = difflib.SequenceMatcher(None, text, ann.text)
                        match = d.find_longest_match(offset, len(text), 0, len(ann.text))
                        begin_string = text[match.a - match.b: match.a + match.b]
                        end_string = text[match.a: match.a + len(ann.text)]
                        begin_count = begin_string.count("\n") - ann.text[:len(begin_string)].count("\n")
                        end_count = end_string.count("\n") - ann.text[len(begin_string):].count("\n")
                        start_position = match.a - (match.b + begin_count)
                        end_position = match.a - match.b + len(ann.text) + end_count
                        ann.start = start_position
                        ann.end = end_position
                        offset = end_position
                        print(text[start_position:end_position] + "\n" + ann.text)
                        ann.text = text[ann.start:ann.end]
                        print("\n")
        except Exception as e:
            print(e)


# def join_dri_annotations_and_texts(texts, annotations, path):
#     # get_char_positions_for_dri_annotations(annotations, path)
#     for doc in texts:
#         token_index = 0
#         for ann in annotations:
#             if doc.file.split('.txt')[0] == ann.file.split('_')[0]:
#                 offset = 0
#                 for i, token in enumerate(doc.tokens):
#                     if not token.matched and i >= token_index:
#                         position = ann.text.find(token.token, offset)
#                         if (position > 0):
#                             offset = position + len(token.token)
#                             token.matched = True
#                             token.sentence_label = ann.category
#                             token_index = i
#                             if (offset >= len(ann.text)):
#                                 token.is_end_of_sentence = True
#                                 # now match the tokens to the sentences ... how to do that?
#                                 # if sentence.token.strip() == ann.text:
#                                 # sentence.matched = True
#                                 # sentence.sentence_label = ann.category
#     # this is just for a sanity check
#     for doc in texts:
#         for token in doc.tokens:
#             if not token.matched:
#                 print("Sentence not matched " + token.token)



def write_conll_format(discourse_ann, path):
    if not os.path.exists(path):
        os.makedirs(path)
    def keyfunc(val):
        return val.file

    doc_groups = []

    for k, g in groupby(discourse_ann, keyfunc):
        doc_groups.append(list(g))

    for doc in doc_groups:
        with codecs.open(os.path.join(path, doc[0].file.split('.txt')[0] + '_conll.txt'), "w", 'utf-8') as target:
            for ann in doc:
                for token in ann.tokens:
                    target.write(str(token.token) + "\t" + str(token.token_label) + "\t" + str(token.sentence_label) + "\n")
                    if token.is_end_of_sentence is True:
                        target.write("\n")
            target.close()


def write_conll_format_multiple(joined_annotations, path):
    if not os.path.exists(path):
        os.makedirs(path)

    def keyfunc_sorting(val):
        return (val.file.split("_")[0], val.start)

    def keyfunc(val):
        return val.file.split("_")[0]

    joined_annotations = sorted(joined_annotations, key=keyfunc_sorting, reverse=False)

    doc_groups = []
    unique_keys = []
    for k, g in groupby(joined_annotations, keyfunc):
        doc_groups.append(list(g))
        unique_keys.append(k)

    for doc in doc_groups:
        with codecs.open(os.path.join(path, doc[0].file.split('_')[0] + '_conll.txt'), "w", 'utf-8') as target:
            for token in doc:
                try:
                    target.write(str(token.token) + "\t" + str(token.token_label) + "\t" + str(token.joined_labels["discourse"]) + "\t" + str(token.joined_labels["aspect"]) + "\t" + str(token.joined_labels["summary"]) +"\n")
                    if token.is_end_of_sentence is True:
                            target.write("\n")
                except Exception as e:
                    print(e)
            target.close()

def write_conll_format_all(joined_annotations, path):
    if not os.path.exists(path):
        os.makedirs(path)

    def keyfunc_sorting(val):
        return (val.file.split("_")[0], val.start)

    def keyfunc(val):
        return val.file.split("_")[0]

    joined_annotations = sorted(joined_annotations, key=keyfunc_sorting, reverse=False)

    doc_groups = []
    unique_keys = []
    for k, g in groupby(joined_annotations, keyfunc):
        doc_groups.append(list(g))
        unique_keys.append(k)

    for doc in doc_groups:
        with codecs.open(os.path.join(path, doc[0].file.split('_')[0] + '_conll.txt'), "w", 'utf-8') as target:
            for token in doc:
                try:
                    target.write(str(token.token) + "\t" + str(token.token_label) + "\t" + str(token.joined_labels["discourse"]) + "\t" + str(token.joined_labels["aspect"]) + "\t" + str(token.joined_labels["summary"]) + "\t" + str(token.joined_labels["citation"]) +"\n")
                    if token.is_end_of_sentence is True:
                            target.write("\n")
                except Exception as e:
                    print(e)
            target.close()




'''Function that should not only tokenize the text but also return the character positions'''
def span_tokenize(text):
    tokens = word_tokenize(text)
    offset = 0
    offset_new = 0
    for token in tokens:
        if token == "\'\'":
            token = '"'
        offset_new = text.find(token, offset)
        if offset_new < 0:
            print("Problem occured with token " + token)
        else:
            offset = offset_new
            yield CoNLL_Token(token, offset, offset+len(token))
            offset += len(token)


'''Function that should not only tokenize the text but also return the character positions for sentences'''
def span_tokenize_sentences(text):
    tokens = sent_tokenize(text)
    offset = 0
    offset_new = 0
    for token in tokens:
        offset_new = text.find(token, offset)
        if offset_new < 0:
            print("Problem occured with token " + token)
        else:
            offset = offset_new
            yield CoNLL_Token(token, offset, offset+len(token))
            offset += len(token)

def create_arg_and_discourse_files():
    argument_annotations = load_annotations("C:/Users/anlausch/Desktop/annotations_arguments_brat/final/compiled_corpus")
    discourse_annotations = load_dri.load_discourse_annotation_as_xml_trees("C:/Users/anlausch/Desktop/DRIcorpus", only_main_categories=True)
    get_char_positions_for_dri_annotations(discourse_annotations, "C:/Users/anlausch/Desktop/annotations_arguments_brat/final/compiled_corpus")
    joined_annotations = join_dri_and_arg_annotations(discourse_annotations, argument_annotations)
    write_conll_format(joined_annotations, "./annotations_conll/")


def main():
    print("Process started")
    argument_annotations = load_annotations("./compiled_corpus")
    discourse_annotations = load_dri.load_discourse_annotation_as_xml_trees("./DRIcorpus")
    aspect_annotations = load_dri.load_aspect_annotation_as_xml_trees("./DRIcorpus")
    summary_annotations = load_dri.load_summary_annotation_as_xml_trees("./DRIcorpus")
    citation_annotations = load_dri.load_citation_purpose_annotation_as_xml_trees("./DRIcorpus", "./compiled_corpus")

    get_char_positions_for_dri_annotations(aspect_annotations, "./compiled_corpus")
    get_char_positions_for_dri_annotations(discourse_annotations,
                                           "./compiled_corpus")
    get_char_positions_for_dri_annotations(summary_annotations,
                                           "./compiled_corpus")

    joined_annotations_discourse = join_dri_and_arg_annotations(discourse_annotations, argument_annotations)
    joined_annotations_summary = join_dri_and_arg_annotations(summary_annotations, argument_annotations)
    joined_annotations_aspect = join_dri_and_arg_annotations(aspect_annotations, argument_annotations)
    joined_annotations_citations = join_dri_and_arg_annotations(citation_annotations, argument_annotations)
    for joined_ann in joined_annotations_citations:
        for i,token in enumerate(joined_ann.tokens):
            token.is_end_of_sentence = False
            if i == 0:
                token.sentence_label = "BEGIN_CIT_CONTEXT"
            else:
                token.sentence_label = "INSIDE_CIT_CONTEXT"
    all = join_all([joined_annotations_aspect, joined_annotations_discourse, joined_annotations_summary, joined_annotations_citations])
    #write_conll_format_multiple(all, "./test")
    write_conll_format_all(all, "./without_abstracts")

    print("Process ended")

if __name__ == "__main__":
    main()