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
import pickle
import os
import csv
from util.PathMux import get_path_to_OS



def _get_IBM_evidence_datset():
    container_file = "IBM_evidence.pkl"
    container_file = get_path_to_OS() +  "/util/pkl/" + container_file
    path = get_path_to_OS() + '/data/IBM/wikipedia_evidence_dataset_29429.csv'

    IBM_corpora = pd.read_csv(path, delimiter=',', index_col=None, header=0,
                                             encoding='unicode_escape')

    return IBM_corpora


def _load_data_and_create_pickle(case_ID=4, data_dir=""):

    pickled = False
    container_file = get_path_to_OS() + "/util/pkl/IBM_evidences.pkl"
    path = data_dir +'/wikipedia_evidence_dataset_29429.csv'

    try:
        file = open(container_file, 'rb')
    except IOError as err:
        pickled = False

    if pickled:
        train = pickle.load(file)
        dev = pickle.load(file)
        test = pickle.load(file)
    else:
        print("Load Dataset from scratch and pickle")
        evidences = pd.read_csv(path, delimiter=',', index_col=None, header=0)
        print(len(evidences))

        np.random.seed(3)
        train, dev, test = np.split(evidences.sample(frac=1),
                                    [int(.49 * len(evidences)), int(.7 * len(evidences))])
        # https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
        # https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
        print(train.shape)
        print(dev.shape)
        print(test.shape)

        # pickle
        fileObject = open(container_file, 'wb')
        pickle.dump(train, fileObject, protocol=2)
        pickle.dump(dev, fileObject, protocol=2)
        pickle.dump(test, fileObject, protocol=2)
        fileObject.close()

        if case_ID == 1:
            return train
        elif case_ID == 2:
            return dev
        elif case_ID == 3:
            return test
        else:
            return None


    container_file = "ISA_data_split_" + str(split_idx) + ".pkl"
    container_file = get_path_to_OS() + "/util/pkl/" + container_file

    # pickle
    fileObject = open(container_file, 'wb')
    pickle.dump(train_data, fileObject, protocol=2)
    pickle.dump(dev_data, fileObject, protocol=2)
    pickle.dump(test_data, fileObject, protocol=2)
    fileObject.close()

    if case_ID == 1:
        return train_data
    elif case_ID == 2:
        return dev_data
    elif case_ID == 3:
        return test_data
    else:
        return None


def get_IBM_evidences_bySet(case=4, data_dir=""):
    container_file = "IBM_evidences.pkl"
    pickled = True
    container_file = get_path_to_OS() + "/util/pkl/" + container_file
    # path = get_path_to_OS()+ '/data/Insufficient_Arg_Support/data-tokenized.tsv'

    try:
        file = open(container_file, 'rb')
    except IOError:
        pickled = False

    if pickled:
        train = pickle.load(file)
        dev = pickle.load(file)
        test = pickle.load(file)
        if case == 1:
            return train
        elif case == 2:
            return dev
        elif case == 3:
            return test
        else:
            return None
    else:
        print("IBM evidence - Load an create pickle..")
        req_data = _load_data_and_create_pickle(case, data_dir)
        return req_data


def _read_tsv_escape(input_file):
    """Reads a tab separated value file."""
    essay_list = pd.read_csv(input_file, delimiter='\t', index_col=None, header=0, encoding='unicode_escape')
    essay_list["ANNOTATION"].fillna("sufficient", inplace=True)
    essay_list["ESSAY_ID"] = [str(entry).zfill(3) for entry in essay_list["ESSAY"]]
    essay_list["ESSAY_ID"] = ["essay" + str(entry) for entry in essay_list["ESSAY"]]

    essay_list["ESSAY_ID"] = essay_list["ESSAY_ID"] + ["_" for i in range(len(essay_list["ESSAY"]))] + essay_list[
        "ARGUMENT"].astype(str)

    # return essay_list.values.tolist()
    return essay_list


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


if __name__ == "__main__":
    #_get_IBM_evidence_datset()
    loaded = get_IBM_evidences_bySet(case=1)
    print(len(loaded))
    loaded = get_IBM_evidences_bySet(case=2)
    print(len(loaded))
    loaded = get_IBM_evidences_bySet(case=3)
    print(len(loaded))
