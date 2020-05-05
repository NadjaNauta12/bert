import glob
import os
import re
import xml.etree.ElementTree as et
from collections import Counter
from enum import Enum
import pickle
import pandas as pd
import xmltodict
import nltk.tokenize
from sklearn.preprocessing import LabelEncoder
import numpy as np
import platform
from sklearn.model_selection import train_test_split
from util.PathMux import get_path_to_OS


def get_ArgRecognition_UGIP_dataset(case_ID=4):
    container_file = "UGIP_splitted.pkl"
    pickled = True
    container_file = get_path_to_OS() + "/util/pkl/" + container_file

    try:
        file = open(container_file, 'rb')
    except IOError as err:
        pickled = False

    if pickled:
        # file = open(container_file, 'rb')
        train = pickle.load(file)
        dev = pickle.load(file)
        test = pickle.load(file)
    else:
        print("Load Dataset for UGIP from scratch and pickle")
        UGIP_complete = _load_ArgRecognition_XML(load_GM=False)
        np.random.seed(4)
        train, dev, test = np.split(UGIP_complete.sample(frac=1),
                                    [int(.49 * len(UGIP_complete)), int(.7 * len(UGIP_complete))])
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


def get_ArgRecognition_GM_dataset(case_ID=4):
    container_file = "GM_splitted.pkl"
    pickled = True
    container_file = get_path_to_OS() + "/util/pkl/" + container_file

    try:
        file = open(container_file, 'rb')
    except IOError as err:
        pickled = False

    if pickled:
        # file = open(container_file, 'rb')
        train = pickle.load(file)
        dev = pickle.load(file)
        test = pickle.load(file)
    else:
        print("Load Dataset for GM from scratch and pickle")
        GM_complete = _load_ArgRecognition_XML(load_GM=True)
        np.random.seed(4)
        train, dev, test = np.split(GM_complete.sample(frac=1),
                                    [int(.49 * len(GM_complete)), int(.7 * len(GM_complete))])

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


def get_ArgRecognition_dataset(case_ID=3):
    if case_ID == 1:
        train_dev = _load_ArgRecognition_XML(True)
        train, dev = train_test_split(train_dev, test_size=0.3)
        print(len(train), len(dev))
        return [train, dev, _load_ArgRecognition_XML(False)]
    if case_ID == 2:
        train_dev = _load_ArgRecognition_XML(False)
        train, dev = train_test_split(train_dev, test_size=0.3)
        print(len(train), len(dev))
        return [train, dev, _load_ArgRecognition_XML(True)]
    else:
        return None


def _load_ArgRecognition_XML(load_GM):
    """EXAMPLE
       <unit id="1arg2">
         <comment>
            <text>I [...] which is  the one that's not natural now?</text>
            <stance>Pro</stance>
         </comment>
         <argument>
            <text>Gay couples should be able to take advantage of the fiscal and legal benefits of marriage</text>
            <stance>Pro</stance>
         </argument>
         <label>3</label>
      </unit>"""
    df_cols = ["id", "comment", "argument", "label"]
    # path = "./data/Argument_Recognition/comarg.v1/comarg"
    path = get_path_to_OS() + "/data/Argument_Recognition/comarg"
    if load_GM:
        path = path + "/GM.xml"
    else:
        path = path + "/UGIP.xml"

    xtree = et.parse(path)
    xroot = xtree.getroot()
    rows = []

    for node in xroot:

        res = [None] * 6
        res[0] = (node.attrib.get(df_cols[0]))
        for child in node:
            # print(child.tag)
            # if child.tag == "unit":
            #     res[0] = child.attrib.get("id")
            #     continue

            if child.tag == "comment":
                for subchild in child.iter():
                    if subchild.tag == "text":
                        res[1] = subchild.text
                    elif subchild.tag == "stance":
                        res[2] = subchild.text
                continue
            if child.tag == "argument":
                for subchild in child.iter():
                    if subchild.tag == "text":
                        res[3] = subchild.text
                    elif subchild.tag == "stance":
                        res[4] = subchild.text
                continue

            if child.tag == "label":
                res[5] = child.text
                continue
        # rows.append(ArgRecognitionEle(res[0], res[1], res[2], res[3], res[4], res[5]))
        rows.append(res)
    print("Content size of xml file", len(rows))
    cols = ["arg_id", "comment_text", "comment_stance", "argument_text", "argument_stance", "label"]
    df = pd.DataFrame.from_records(rows, columns=cols)
    """Label Mapping:  A is 1 , S is 5 """
    # print(df.label.value_counts())

    """Check Baseline for two additional Tasks"""
    # fn = lambda row: int(1) if row.label == "1" or row.label == "2" else (
    #     int(3) if row.label == "5" or row.label == "4" else int(2))
    # if additional_tasks:
    #     # do same but attach it to the dataframe
    #     df['task_3labels'] = df.apply(fn, axis=1)

    return df


if __name__ == "__main__":
    loaded = get_ArgRecognition_UGIP_dataset(case_ID=1)
    print(len(loaded))
    loaded = get_ArgRecognition_UGIP_dataset(case_ID=2)
    print(len(loaded))
    loaded = get_ArgRecognition_UGIP_dataset(case_ID=3)
    print(len(loaded))

    loaded = get_ArgRecognition_GM_dataset(case_ID=1)
    print(len(loaded))
    loaded = get_ArgRecognition_GM_dataset(case_ID=2)
    print(len(loaded))
    loaded = get_ArgRecognition_GM_dataset(case_ID=3)
    print(len(loaded))
