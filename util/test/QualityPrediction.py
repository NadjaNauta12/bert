import pandas as pd
import glob

path = r"C:\Users\Wifo\Documents\Universität Mannheim\Master\Masterthesis\Datasets\Argument Quality\IBM-ArgQ-9.1kPairs"

all_files = glob.glob(path + "/*.csv")

print(len(all_files))

read_files_single = []
for filename in all_files:
    # df = pd.read_csv(filename, delimiter='\t', index_col=None, header=0)
    df = pd.read_csv(filename, delimiter='\t', index_col=None, header=0)
    read_files_single.append(df)
    # print("hello")

quality_corpus = pd.concat(read_files_single, axis=0, ignore_index=True)
for col in quality_corpus.columns:
    print(col)
#print(quality_corpus.head())
print(quality_corpus[:1])

print(quality_corpus.describe())

print("Unique values for label \n", quality_corpus.label.unique())
print("Unique values for label \n", quality_corpus.label.value_counts())
print("unique values for annotatorid \n", quality_corpus.annotatorid.unique())
# print("unique values for id \n", quality_corpus.id.unique())
print("How many unique ones \n", len(quality_corpus.loc[: ,"#id"].unique()))

