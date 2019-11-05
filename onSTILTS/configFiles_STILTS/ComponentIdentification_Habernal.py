import glob
from xml.etree import ElementTree

import pandas as pd
import xml.etree.ElementTree


path = r"C:\Users\Wifo\Documents\Universität Mannheim\Master\Masterthesis\Datasets\Argument Component Identification Habernal\brat-project-final"

all_files = glob.glob(path + "/*.ann")
col_names = ["Tag", "Identifiers", "Explanation"]
print(len(all_files))

read_ANNfiles_single = []
for filename in all_files:
    # df = pd.read_csv(filename, delimiter='\t', index_col=None, header=0)
    df = pd.read_csv(filename, delimiter='\t', header=None, names=col_names)
    read_ANNfiles_single.append(df)
    # print("hello")

ACI_ANN_Habernal_corpus = pd.concat(read_ANNfiles_single, axis=0, ignore_index=True)
for col in ACI_ANN_Habernal_corpus.columns:
    print(col)


print("First entry\n",ACI_ANN_Habernal_corpus[:1])

print("Statistics\n", ACI_ANN_Habernal_corpus.describe())


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
    lines = [l.decode('utf8', 'ignore')for l  in f.readlines()]



col_names= ["Essay", "Prompts"]
prompts = pd.DataFrame(lines,  columns=col_names)
print(prompts.head())
print("#prompts:\n", len(prompts))

print("Statistics\n", prompts.describe())




