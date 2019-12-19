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
#from sklearn.preprocessing import LabelEncoder
import os
import platform

def get_ArgZoning_dataset(case=4):
    """
    Adapted from A. Lauscher

    """
    if os.name == "nt":
        path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Zoning"
    elif platform.release() != "4.9.0-11-amd64":  # GOOGLE COLAB
        print("AQ_Google Colab")
        path = "/content/drive/My Drive/Masterthesis/data/Argument_Zoning"
    else:
        path = "/work/nseemann/data/Argument_Zoning"

    if case == 1:
        path = path + "/train"
    elif case == 2:
        path = path + "/dev"
    elif case == 3:
        path = path + "/test"
    else:
        path = path
    documents = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            sentences = []
            if '.az-scixml' in file:
                tree = et.parse(os.path.join(subdir, file))
                for elem in tree.iter():
                    if elem.tag in ['S'] and elem.get("AZ") is not None:
                        elem_string = et.tostring(elem)#, encoding="unicode") # TODO MARKER -  Problems with Server
                        clean_string = re.sub(r'<\/?S[^>]*>', '', elem_string)
                        clean_string = re.sub(r'<\/?REFAUTHOR[^>]*>', '', clean_string)
                        clean_string = re.sub(r'<\/?REF[^>]*>', '', clean_string)
                        clean_string = re.sub(r'<\/?CREF[^>]*>', '</NUM>', clean_string)
                        sentence = Sentence(text=clean_string.strip(), category=elem.get("AZ"),
                                            file=file, sentence_id=elem.get("ID"))
                        sentences.append(sentence)
                documents.append(sentences)
        print("Number of documents loaded: ", len(documents))
        occurrences = Counter([sentence for sentences in documents for sentence in sentences])
        # print("Stats: ", occurrences)

        df = pd.DataFrame([sentence.__dict__ for sentences in documents for sentence in sentences])

        #LE = LabelEncoder()
        #df['target_label'] = LE.fit_transform(df['AZ_category'])
        print(df['AZ_category'].unique())
        # print(df.head())
        # variables = arr[0].keys()
        # df = pd.DataFrame([[getattr(i, j) for j in variables] for i in arr], columns=variables)

        # df = pd.DataFrame.from_records(documents)

        return df


class Sentence:
    def __init__(self, text, category, file, sentence_id):
        self.text = text
        self.AZ_category = category
        self.file = file
        self.sentenceID = sentence_id


if __name__ == "__main__":
    loaded = get_ArgZoning_dataset(case=1)
    print(loaded.shape)
    loaded = get_ArgZoning_dataset(case=2)
    print(loaded.shape)
    loaded = get_ArgZoning_dataset(case=3)
    print(loaded.shape)
