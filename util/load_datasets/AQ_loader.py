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
import platform

def get_QualityPrediction_dataset():
    print("Loading Argument Quality Prediction Dataset")
    return [load_QualityPrediction_datset_2part(test_set=False), load_QualityPrediction_datset_2part(test_set=True)]


def load_QualityPrediction_datset_2part(test_set):
    # print("Loading Argument Quality Prediction Dataset")
    if test_set:
        path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Quality\9.1_test"
        all_files = glob.glob(path + "/*.csv")
    else:
        path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Quality\9.1_train_dev"
        all_files = glob.glob(path + "/*.csv")

    # print(len(all_files))
    read_files_single = []
    for filename in all_files:
        # df = pd.read_csv(filename, delimiter='\t', index_col=None, header=0)
        df = pd.read_csv(filename, delimiter='\t', index_col=None, header=0)
        read_files_single.append(df)

    quality_corpus = pd.concat(read_files_single, axis=0, ignore_index=True)
    # quality_corpus= pd.DataFrame(quality_corpus)
    # print(quality_corpus.label.unique())
    # print(quality_corpus.annotatorid.unique())
    print(quality_corpus['label'].value_counts())
    fn = lambda row: 1 if row.label == "a1" else 0
    quality_corpus['target_label'] = quality_corpus.apply(fn, axis=1)
    return quality_corpus


def load_ArgQuality_datset(case_ID=4):
    """

    :param case_ID: 1 fro train, 2 for dev, 3 for test
    :return:
    """
    pickled = True
    if os.name == "nt":
        container_file = r"C:\Users\Wifo\PycharmProjects\Masterthesis\util\AQ_data.pkl"
        path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Quality\IBM-ArgQ-9.1kPairs"
    elif platform.release() != "4.9.0-11-amd64":  # GOOGLE COLAB
        print("AQ_Google Colab")
        container_file = "/content/bert/pkl/util/pkl/AQ_data.pkl"
        path = "/content/drive/My Drive/Masterthesis/data/Argument_Quality/IBM-ArgQ-9.1kPairs"
    else:
        container_file = "/work/nseemann/util/pkl/AQ_data.pkl"
        path = "/work/nseemann/data/Argument_Quality/IBM-ArgQ-9.1kPairs"

    try:
        file = open(container_file, 'rb')
    except IOError as err:
        pickled = False
    pickled = False
    if pickled:
        # file = open(container_file, 'rb')
        train = pickle.load(file)
        dev = pickle.load(file)
        test = pickle.load(file)
    else:
        print("Load Dataset from scratch and pickle")
        all_files = glob.glob(path + "/*.csv")
        read_files_single = []
        for filename in all_files:
            # df = pd.read_csv(filename, delimiter='\t', index_col=None, header=0)
            df = pd.read_csv(filename, delimiter='\t', index_col=None, header=0)
            df['file_id'] = filename.split('\\')[len(filename.split('\\')) - 1]
            read_files_single.append(df)

        quality_corpus = pd.concat(read_files_single, axis=0, ignore_index=True)
        # print(quality_corpus['label'].value_counts())
        #fn = lambda row: 1 if row.label == "a1" else 0
        #quality_corpus['target_label'] = quality_corpus.apply(fn, axis=1)
        print(len(quality_corpus))
        np.random.seed(3)
        train, dev, test = np.split(quality_corpus.sample(frac=1),
                                    [int(.49 * len(quality_corpus)), int(.7 * len(quality_corpus))])
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


if __name__ == "__main__":
    loaded_train = load_ArgQuality_datset(case_ID=1)
    print(loaded_train.head())
