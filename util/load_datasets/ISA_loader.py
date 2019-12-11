import glob
import os
import re
import xml.etree.ElementTree as et
from collections import Counter
from enum import Enum
import numpy as np
import pandas as pd
import xmltodict
import nltk.tokenize
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import tensorflow as tf
import csv


def _get_InsuffientSupport_datset():
    container_file = "ISA_data.pkl"
    if os.name == "nt":
        container_file = "C:/Users/Wifo/PycharmProjects/Masterthesis/util/" + container_file
        path = 'C:/Users/Wifo/PycharmProjects/Masterthesis/data/Insufficient_Arg_Support/data-tokenized.tsv'
    elif os.name == "posix":  # GOOGLE COLAB
        print("AQ_Google Colab")
        container_file = "/content/bert/util/" + container_file
        path = "/content/drive/My Drive/Masterthesis/data/Insufficient_Arg_Support/data-tokenized.tsv"
    else:
        container_file = "/work/nseemann/util/" + container_file
        path = "/work/nseemann/data/Insufficient_Arg_Support/data-tokenized.tsv"

    insufficientSupper_corpora = pd.read_csv(path, delimiter='\t', index_col=None, header=0,
                                             encoding='unicode_escape')
    insufficientSupper_corpora["ANNOTATION"].fillna("sufficient", inplace=True)
    insufficientSupper_corpora["ESSAY_ID"] = ["essay" + str(entry).zfill(3) for entry in
                                              insufficientSupper_corpora["ESSAY"]]
    insufficientSupper_corpora["ESSAY_ID"] = insufficientSupper_corpora["ESSAY_ID"] \
                                             + ["_" for i in range(len(insufficientSupper_corpora["ESSAY"]))] \
                                             + insufficientSupper_corpora[
                                                 "ARGUMENT"].astype(str)

    return insufficientSupper_corpora


def read_data_splitting(idx):
    if os.name == "nt":
        path = 'C:/Users/Wifo/PycharmProjects/Masterthesis/data/Insufficient_Arg_Support/data-tokenized.tsv'
    elif os.name == "posix":  # GOOGLE COLAB
        print("AQ_Google Colab")
        path = "/content/drive/My Drive/Masterthesis/data/Insufficient_Arg_Support/data-tokenized.tsv"
    else:
        path = "/work/nseemann/data/Insufficient_Arg_Support/data-tokenized.tsv"

    complete_split = _read_tsv(input_file=path)
    # print(type(complete_split))
    # print(len(complete_split))
    # print(complete_split[1])
    # print(len(complete_split[1]))
    complete_split = np.array(complete_split)
    cols = list(range(1, 102))
    # cols = ["Essay_ID"] + cols
    # print(cols)
    # print(complete_split[0: , 1:].shape)
    df_split = pd.DataFrame(data=complete_split[0:, 1:], index=complete_split[0:, 0], columns=cols)
    # print(df_split.shape)
    # print(df_split.head())
    df_split.drop(df_split.columns[len(df_split.columns) - 1], axis=1, inplace=True)
    # print(df_split.head())
    self._data_split = df_split

    train = []
    test = []
    dev = []
    index = self._data_split.index.values
    for i in range(0, self._data_split.shape[0]):
        cur = self._data_split[idx][i]
        if cur == "TRAIN":
            train.append(index[i].strip())
        elif cur == "DEV":
            dev.append(index[i].strip())
        elif cur == "TEST":
            test.append(index[i].strip())
        else:
            print("something went wrong")
            raise NotImplementedError

    return [train, dev, test]

    def __get_examples(self, filter_list, descr):
        insufficient_corpus = data_loader.get_InsuffientSupport_datset()
        filtered = insufficient_corpus.loc[insufficient_corpus["ESSAY_ID"].isin(filter_list)]
        print(filtered[:1])
        print(len(filtered))
        dev_InputExamples = self.convert_To_InputExamples(filtered, descr)
        print("Convert Data to InputExample")
        return dev_InputExamples

    def _read_tsv_escape(cls, input_file, quotechar=None, errors='ignore'):
        """Reads a tab separated value file."""
        essay_list = pd.read_csv(input_file, delimiter='\t', index_col=None, header=0, encoding='unicode_escape')
        essay_list["ANNOTATION"].fillna("sufficient", inplace=True)
        essay_list["ESSAY_ID"] = [str(entry).zfill(3) for entry in essay_list["ESSAY"]]
        essay_list["ESSAY_ID"] = ["essay" + str(entry) for entry in essay_list["ESSAY"]]

        essay_list["ESSAY_ID"] = essay_list["ESSAY_ID"] + ["_" for i in range(len(essay_list["ESSAY"]))] + essay_list[
            "ARGUMENT"].astype(str)

        # return essay_list.values.tolist()
        return essay_list
        # with tf.gfile.Open(input_file, "r") as f:
        #
        #     try:
        #         reader = csv.reader(f, delimiter="\t", quotechar=quotechar, encoding='utf-16')
        #     except csv.Error as e:
        #         print("Failed to read file at line", reader.file_num)
        #     lines = []
        #     for line in reader:
        #         lines.append(line)
        #     return lines


def _load_data_and_create_pickle(split_idx, case=4):
    train, dev, test = read_data_splitting(split_idx)
    insufficient_corpus = _get_InsuffientSupport_datset()
    train_data = insufficient_corpus.loc[insufficient_corpus["ESSAY_ID"].isin(train)]
    dev_data = insufficient_corpus.loc[insufficient_corpus["ESSAY_ID"].isin(dev)]
    test_data = insufficient_corpus.loc[insufficient_corpus["ESSAY_ID"].isin(test)]

    if os.name == "nt":
        container_file = "C:/Users/Wifo/PycharmProjects/Masterthesis/util/" + container_file
    elif os.name == "posix":  # GOOGLE COLAB
        print("AQ_Google Colab")
        container_file = "/content/bert/util/" + contailer_file
    else:
        container_file = "/work/nseemann/util/" + contailer_file

    # pickle
    fileObject = open(container_file, 'wb')
    pickle.dump(train, fileObject)
    pickle.dump(dev, fileObject)
    pickle.dump(test, fileObject)
    fileObject.close()

    if case_ID == 1:
        return train_data
    elif case_ID == 2:
        return dev_data
    elif case_ID == 3:
        return test_data
    else:
        return None


def get_InsuffientSupport_datset_byFilter(split_idx=1, case=4):
    container_file = "ISA_data_split" + str(split_idx) + ".pkl"
    pickled = True
    if os.name == "nt":
        container_file = "C:/Users/Wifo/PycharmProjects/Masterthesis/util/" + container_file
        path = 'C:/Users/Wifo/PycharmProjects/Masterthesis/data/Insufficient_Arg_Support/data-tokenized.tsv'
    elif os.name == "posix":  # GOOGLE COLAB
        print("AQ_Google Colab")
        container_file = "/content/bert/util/" + container_file
        path = "/content/drive/My Drive/Masterthesis/data/Insufficient_Arg_Support/data-tokenized.tsv"
    else:
        container_file = "/work/nseemann/util/" + container_file
        path = "/work/nseemann/data/Insufficient_Arg_Support/data-tokenized.tsv"

    try:
        file = open(container_file, 'rb')
    except FileNotFoundError as err:
        pickled = False

    if pickled:
        train = pickle.load(file)
        dev = pickle.load(file)
        test = pickle.load(file)
        if case_ID == 1:
            return train
        elif case_ID == 2:
            return dev
        elif case_ID == 3:
            return test
        else:
            return None
    else:
        print("ISA - Load an create pickle..")
        req_data = _load_data_and_create_pickle(split_idx, case)
        return req_data


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

if __name__ == "__main__":
    loaded = get_InsuffientSupport_datset_byFilter(case=1)
    print(len(loaded))
    loaded = get_InsuffientSupport_datset_byFilter(case=2)
    print(len(loaded))
    loaded = get_InsuffientSupport_datset_byFilter(case=3)
    print(len(loaded))
