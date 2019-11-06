from util.load_datasets.data_loader import *
import pickle
import numpy as np
import pandas as pd

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report


def majority_vote(data, columnname, task, multilabel=None):
    print(task, "BASELINE - Majority vote baseline running")

    major_label = data[columnname].mode()[0]
    Print("Major label is:", major_label)
    # print(major_label)
    # print(type(major_label))
    predictions = []
    for row in range(data.shape[0]):
        predictions.append(major_label)
    # print(len(predictions), len(data.ANNOTATION.tolist()))
    # df = pd.DataFrame(predictions, columns= ["huhu"])
    # print(  print(df.isna().sum()))
    result = Result(label_list=data[columnname].unique(), predictions=predictions, truth=data[columnname].tolist(),
                    descr="BL Majority Vote - InsufficientSupport")
    # pickle.dump(result, open("./results/Baseline_MV_InsufficientSupport.txt", 'wb'))
    if multilabel is None:
        # label =[1,2,3,4,5]
        eva = classification_report(result.truth, result.predictions, labels=result.label_list, digits=3)
    else:
        eva = classification_report(result.truth, result.predictions, labels=multilabel, digits=3)

    print("Evaluation result MV:")
    print(eva)
    print("Done")


class Result:
    def __init__(self, label_list, predictions, truth, descr):
        self.label_list = label_list
        self.predictions = predictions
        self.truth = truth
        self.description = descr


def main():
    # corpora = get_InsuffientSupport_datset()
    # majority_vote(corpora, "ANNOTATION", "Insufficient Supported Arguments")

    # corpora = get_QualityPrediction_dataset()
    # majority_vote(corpora, "label", "Argument Quality Prediction")

    # corpora = get_ArgRecognition_dataset()
    multilabel_ArgRec = [1, 2, 3, 4, 5]
    # majority_vote(corpora[0], "label", "Argument Recogniiton GM", multilabel=multilabel_ArgRec)
    # majority_vote(corpora[1], "label", "Argument Recognition UGIP", multilabel=multilabel_ArgRec)
    # corpora_whole = corpora[0].append(corpora[1], ignore_index=True)
    # print(corpora_whole.head())
    # print(corpora_whole.describe())
    # print(corpora_whole.groupby("comment_text").head())
    # print(corpora_whole.label.value_counts())
    # majority_vote(corpora_whole, "label", "Argument Recognition  GM & UGIP", multilabel=multilabel_ArgRec)
    print("hello")
    corpora = get_ArgZoning_dataset()
    multilabel_AZI = corpora.AZ_category.unique()
    majority_vote(corpora, "AZ_category", "Argument Zoning I", multilabel=multilabel_AZI)

    ##### NOT DONE YET
    # corpora = get_ArgZoning_dataset()
    # majority_vote(corpora, "ANNOTATION", "Insufficient Supported Arguments")


if __name__ == '__main__':
    main()
