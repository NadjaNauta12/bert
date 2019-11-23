import pandas as pd
pd.set_option('display.width', 400)
pd.set_option('display.max_columns',10)
path = r"C:\Users\Wifo\Documents\Universität Mannheim\Master\Masterthesis\Datasets\Insufficient Arg Support"
insufficientSupper_corpora = pd.read_csv(path + "\data-tokenized.tsv", delimiter='\t', index_col=None, header=0, encoding = 'unicode_escape')
print("Colums of the corpus\t")
for col in insufficientSupper_corpora.columns:
    print(col)
insufficientSupper_corpora["ANNOTATION"].fillna("sufficient", inplace=True)

print(insufficientSupper_corpora.head())
#print(insufficientSupper_corpora[:3])


print(insufficientSupper_corpora.describe(include='all'))

#print("How many essays:\n", insufficientSupper_corpora.ESSAY.value_counts())
print("Unique values for annotation \n", insufficientSupper_corpora.ANNOTATION.value_counts())
#print("unique values for annotatorid \n", insufficientSupper_corpora.annotatorid.unique())
# print("unique values for id \n", insufficientSupper_corpora.id.unique())
#print("How many unique ones \n", len(insufficientSupper_corpora.loc[: ,"#id"].unique()))

