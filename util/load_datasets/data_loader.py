import glob
import os
import re
import xml.etree.ElementTree as et
from collections import Counter
from enum import Enum

import pandas as pd
import xmltodict
import nltk.tokenize
from sklearn.preprocessing import LabelEncoder


def gti get_InsuffientSupport_datset():
    path = 'C:/Users/Wifo/PycharmProjects/Masterthesis/data/Insufficient_Arg_Support/data-tokenized.tsv'
    insufficientSupper_corpora = pd.read_csv(path, delimiter='\t', index_col=None, header=0, encoding='unicode_escape')
    insufficientSupper_corpora["ANNOTATION"].fillna("sufficient", inplace=True)
    # insufficientSupper_corpora["ESSAY_ID"] = [str(entry).zfill(3) for entry in insufficientSupper_corpora["ESSAY"]]
    insufficientSupper_corpora["ESSAY_ID"] = ["essay" + str(entry).zfill(3) for entry in insufficientSupper_corpora["ESSAY"]]
    insufficientSupper_corpora["ESSAY_ID"] = insufficientSupper_corpora["ESSAY_ID"] \
                                             + ["_" for i in range(len(insufficientSupper_corpora["ESSAY"]))] \
                                             + insufficientSupper_corpora[
                                                 "ARGUMENT"].astype(str)

    return insufficientSupper_corpora

#
# def get_QualityPrediction_dataset():
#     print("Loading Argument Quality Prediction Dataset")
#     return [load_QualityPrediction_datset(test_set=False), load_QualityPrediction_datset(test_set=True)]
#
#
# def load_QualityPrediction_datset(test_set):
#     #print("Loading Argument Quality Prediction Dataset")
#     if test_set:
#         path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Quality\9.1_test"
#         all_files = glob.glob(path + "/*.csv")
#     else:
#         path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Quality\9.1_train_dev"
#         all_files = glob.glob(path + "/*.csv")
#
#     # print(len(all_files))
#     read_files_single = []
#     for filename in all_files:
#         # df = pd.read_csv(filename, delimiter='\t', index_col=None, header=0)
#         df = pd.read_csv(filename, delimiter='\t', index_col=None, header=0)
#         read_files_single.append(df)
#
#     quality_corpus = pd.concat(read_files_single, axis=0, ignore_index=True)
#     # quality_corpus= pd.DataFrame(quality_corpus)
#     # print(quality_corpus.label.unique())
#     # print(quality_corpus.annotatorid.unique())
#     print(quality_corpus['label'].value_counts())
#     fn = lambda row: 1 if row.label == "a1" else 0
#     quality_corpus['target_label'] = quality_corpus.apply(fn, axis=1)
#     return quality_corpus
#
# def get_ArgRecognition_UGIP_dataset():
#     return _load_ArgRecognition_XML(load_GM=False, additional_tasks=False)
#
#
# def get_ArgRecognition_GM_dataset():
#     return _load_ArgRecognition_XML(load_GM=True, additional_tasks=False)
#
#
# def get_ArgRecognition_dataset(additional_tasks=False):
#     return [_load_ArgRecognition_XML(True, additional_tasks), _load_ArgRecognition_XML(False, additional_tasks)]
#
#
# def _load_ArgRecognition_XML(load_GM, additional_tasks):
#     """EXAMPLE
#        <unit id="1arg2">
#          <comment>
#             <text>I [...] which is  the one that's not natural now?</text>
#             <stance>Pro</stance>
#          </comment>
#          <argument>
#             <text>Gay couples should be able to take advantage of the fiscal and legal benefits of marriage</text>
#             <stance>Pro</stance>
#          </argument>
#          <label>3</label>
#       </unit>"""
#     df_cols = ["id", "comment", "argument", "label"]
#     # path = "./data/Argument_Recognition/comarg.v1/comarg"
#     path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Recognition\comarg.v1\comarg"
#     if load_GM:
#         path = path + "/GM.xml"
#     else:
#         path = path + "/UGIP.xml"
#
#     xtree = et.parse(path)
#     xroot = xtree.getroot()
#     rows = []
#
#     for node in xroot:
#
#         res = [None] * 6
#         res[0] = (node.attrib.get(df_cols[0]))
#         for child in node:
#             # print(child.tag)
#             # if child.tag == "unit":
#             #     res[0] = child.attrib.get("id")
#             #     continue
#
#             if child.tag == "comment":
#                 for subchild in child.iter():
#                     if subchild.tag == "text":
#                         res[1] = subchild.text
#                     elif subchild.tag == "stance":
#                         res[2] = subchild.text
#                 continue
#             if child.tag == "argument":
#                 for subchild in child.iter():
#                     if subchild.tag == "text":
#                         res[3] = subchild.text
#                     elif subchild.tag == "stance":
#                         res[4] = subchild.text
#                 continue
#
#             if child.tag == "label":
#                 res[5] = child.text
#                 continue
#         # rows.append(ArgRecognitionEle(res[0], res[1], res[2], res[3], res[4], res[5]))
#         rows.append(res)
#     print("Content size of xml file", len(rows))
#     cols = ["arg_id", "comment_text", "comment_stance", "argument_text", "argument_stance", "label"]
#     df = pd.DataFrame.from_records(rows, columns=cols)
#     """Label Mapping:  A is 1 , S is 5 """
#     # print(df.label.value_counts())
#
#     """Check Baseline for two additional Tasks"""
#     fn = lambda row: int(1) if row.label == "1" or row.label == "2" else (
#         int(3) if row.label == "5" or row.label == "4" else int(2))
#     if additional_tasks:
#         # do same but attach it to the dataframe
#         df['task_3labels'] = df.apply(fn, axis=1)
#
#     return df

#
# def get_ArgZoning_dataset(path=r'./data/Argument_Zoning'):
#     """
#     Adapted from A. Lauscher
#
#     """
#     #path = r'./data/Argument_Zoning'
#     documents = []
#     for subdir, dirs, files in os.walk(path):
#         for file in files:
#             sentences = []
#             if '.az-scixml' in file:
#                 tree = et.parse(os.path.join(subdir, file))
#                 for elem in tree.iter():
#                     if elem.tag in ['S'] and elem.get("AZ") is not None:
#                         elem_string = et.tostring(elem, encoding="unicode")
#                         clean_string = re.sub(r'<\/?S[^>]*>', '', elem_string)
#                         clean_string = re.sub(r'<\/?REFAUTHOR[^>]*>', '', clean_string)
#                         clean_string = re.sub(r'<\/?REF[^>]*>', '', clean_string)
#                         clean_string = re.sub(r'<\/?CREF[^>]*>', '</NUM>', clean_string)
#                         sentence = Sentence(text=clean_string.strip(), category=elem.get("AZ"),
#                                             file=file, sentence_id=elem.get("ID"))
#                         sentences.append(sentence)
#                 documents.append(sentences)
#         print("Number of documents loaded: ", len(documents))
#         occurrences = Counter([sentence for sentences in documents for sentence in sentences])
#         #print("Stats: ", occurrences)
#
#         df = pd.DataFrame([sentence.__dict__ for sentences in documents for sentence in sentences])
#
#         LE = LabelEncoder()
#         df['target_label'] = LE.fit_transform(df['AZ_category'])
#         #print(df['AZ_category'].unique())
#        # print(df.head())
#         # variables = arr[0].keys()
#         # df = pd.DataFrame([[getattr(i, j) for j in variables] for i in arr], columns=variables)
#
#         # df = pd.DataFrame.from_records(documents)
#
#         return df


def get_ACI_dataset_Habernal():
    path = './data/Argument_Component_Identification_Habernal/brat-project-final'
    all_files = glob.glob(path + "/*.ann")
    col_names = ["Tag", "Identifiers", "Explanation"]
    # print(len(all_files))

    read_ANNfiles_single = []
    for filename in all_files:
        # df = pd.read_csv(filename, delimiter='\t', index_col=None, header=0)
        df = pd.read_csv(filename, delimiter='\t', header=None, names=col_names)
        read_ANNfiles_single.append(df)

    ACI_ANN_Habernal_corpus = pd.concat(read_ANNfiles_single, axis=0, ignore_index=True)

    all_files = glob.glob(path + "/*.txt")
    col_names = ["Document"]
    print(len(all_files))

    read_Documents_single = []
    for filename in all_files:
        df = pd.read_csv(filename, names=col_names)
        read_Documents_single.append(df)

    print("#files:\n", len(read_Documents_single))

    ACI_DOC_Habernal_corpus = pd.concat(read_Documents_single, axis=0, ignore_index=True)
    print("Statistics\n", ACI_DOC_Habernal_corpus.describe())

    path = r"C:\Users\Wifo\Documents\Universität Mannheim\Master\Masterthesis\Datasets\Argument Component Identification Habernal"
    # get the prompts
    with open(path + "/prompts.csv", 'rb') as f:
        lines = [l.decode('utf8', 'ignore') for l in f.readlines()]

    col_names = ["Essay", "Prompts"]
    prompts = pd.DataFrame(lines, columns=col_names)
    print(prompts.head())
    print("#prompts:\n", len(prompts))
    print("Statistics\n", prompts.describe())
    return ACI_ANN_Habernal_corpus, ACI_DOC_Habernal_corpus, prompts


def get_ACI_dataset_Lauscher():
    pass


class ArgRecognitionEle():

    def __init__(self, arg_id, comment_text, comment_stance, argument_text, argument_stance, label):
        self.id = arg_id
        self.comment_text = comment_text
        self.comment_stance = comment_stance
        self.argument_text = argument_text
        self.argument_stance = argument_stance
        self.label = label


class Sentence:
    def __init__(self, text, category, file, sentence_id):
        self.text = text
        self.AZ_category = category
        self.file = file
        self.sentenceID = sentence_id


# '''Simple class for representing the desired output'''
# class CoNLL_Token:
#     def __init__(self, token, start, end, token_label=None, sentence_label=None, is_end_of_sentence=False, file=None):
#         self.token = token
#         self.start = start
#         self.end = end
#         if token_label is not None:
#             self.token_label = token_label
#         else:
#             self.token_label = Token_Label.OUTSIDE
#         self.sentence_label = sentence_label
#         self.matched = False
#         self.is_end_of_sentence = is_end_of_sentence
#         self.file = file

'''Enum for representing our argument labels'''


class Token_Label(Enum):
    BEGIN_BACKGROUND_CLAIM = 1
    INSIDE_BACKGROUND_CLAIM = 2
    BEGIN_OWN_CLAIM = 3
    INSIDE_OWN_CLAIM = 4
    BEGIN_DATA = 5
    INSIDE_DATA = 6
    OUTSIDE = 7
