import glob
from xml.etree import ElementTree

import pandas as pd
import xml.etree.ElementTree


path = r"C:\Users\Wifo\Documents\Universität Mannheim\Master\Masterthesis\Datasets\Argument Component Identification Lauscher\compiled_corpus"

all_files = glob.glob(path + "/*.ann")
col_names = ["Tag", "Identifiers", "Explanation"]
print(len(all_files))

read_ANNfiles_single = []
for filename in all_files:
    # df = pd.read_csv(filename, delimiter='\t', index_col=None, header=0)
    df = pd.read_csv(filename, delimiter='\t', header=None, names=col_names)
    read_ANNfiles_single.append(df)
    # print("hello")

ACI_ANN_Lauscher_corpus = pd.concat(read_ANNfiles_single, axis=0, ignore_index=True)
for col in ACI_ANN_Lauscher_corpus.columns:
    print(col)
#print(quality_corpus.head())
print("First Entry \n", ACI_ANN_Lauscher_corpus[:1])

print("Description of Corpus\n", ACI_ANN_Lauscher_corpus.describe())




