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


def get_InsuffientSupport_datset():
    path = './data/Insufficient_Arg_Support/data-tokenized.tsv'
    insufficientSupper_corpora = pd.read_csv(path, delimiter='\t', index_col=None, header=0, encoding='unicode_escape')
    insufficientSupper_corpora["ANNOTATION"].fillna("sufficient", inplace=True)
    # insufficientSupper_corpora["ESSAY_ID"] = [str(entry).zfill(3) for entry in insufficientSupper_corpora["ESSAY"]]
    insufficientSupper_corpora["ESSAY_ID"] = ["essay" + str(entry).zfill(3) for entry in insufficientSupper_corpora["ESSAY"]]
    insufficientSupper_corpora["ESSAY_ID"] = insufficientSupper_corpora["ESSAY_ID"] \
                                             + ["_" for i in range(len(insufficientSupper_corpora["ESSAY"]))] \
                                             + insufficientSupper_corpora[
                                                 "ARGUMENT"].astype(str)

    return insufficientSupper_corpora


def get_ACI_dataset_Stab():
    path = './data/Argument_Component_Identification_Stab/brat-project-final'
    all_files = glob.glob(path + "/*.ann")
    col_names = ["Tag", "Identifiers", "Explanation"]
    # print(len(all_files))

    read_ANNfiles_single = []
    for filename in all_files:
        # df = pd.read_csv(filename, delimiter='\t', index_col=None, header=0)
        df = pd.read_csv(filename, delimiter='\t', header=None, names=col_names)
        read_ANNfiles_single.append(df)

    ACI_ANN_Stab_corpus = pd.concat(read_ANNfiles_single, axis=0, ignore_index=True)

    all_files = glob.glob(path + "/*.txt")
    col_names = ["Document"]
    print(len(all_files))

    read_Documents_single = []
    for filename in all_files:
        df = pd.read_csv(filename, names=col_names)
        read_Documents_single.append(df)

    print("#files:\n", len(read_Documents_single))

    ACI_DOC_Stab_corpus = pd.concat(read_Documents_single, axis=0, ignore_index=True)
    print("Statistics\n", ACI_DOC_Stab_corpus.describe())

    path = "./data/Argument_Component_Identification_Stab"
    # get the prompts
    with open(path + "/prompts.csv", 'rb') as f:
        lines = [l.decode('utf8', 'ignore') for l in f.readlines()]

    col_names = ["Essay", "Prompts"]
    prompts = pd.DataFrame(lines, columns=col_names)
    print(prompts.head())
    print("#prompts:\n", len(prompts))
    print("Statistics\n", prompts.describe())
    return ACI_ANN_Stab_corpus, ACI_DOC_Stab_corpus, prompts


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
