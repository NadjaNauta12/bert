import pandas as pd
import os.path
import toolkit
from datetime import datetime
from sklearn.metrics import classification_report
from custom_exceptions import EvaluationFileAlreadyExists
from load_datasets import brat_annotations
from util.load_datasets.data_loader import *
import pickle
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)


def majority_vote(train, test, column, task, multilabel=None):
    print(task, "BASELINE - Majority vote baseline running")

    major_label = train[column].mode()[0]
    print("Major label is:", major_label)
    # print(major_label)
    # print(type(major_label))
    predictions = []
    for row in range(test.shape[0]):
        predictions.append(major_label)
    # print(len(predictions), len(data.ANNOTATION.tolist()))
    # df = pd.DataFrame(predictions, columns= ["huhu"])
    # print(  print(df.isna().sum()))
    result = Result(label_list=train[column].unique(), predictions=predictions, truth=test[column].tolist(),
                    descr="BL Majority Vote " + task)
    # pickle.dump(result, open("./results/Baseline_MV_InsufficientSupport.txt", 'wb'))

    if multilabel is None:
        # label =[1,2,3,4,5]
        eva = classification_report(result.truth, result.predictions, labels=result.label_list, digits=3)
        """ ROC AUC"""
        roc_auc = roc_auc_score(y_true=result.truth, y_score=result.predictions, average="macro", sample_weight=None,
                            max_fpr=None)
        print("ROC: ", roc_auc)
    else:
        print("Type: ", type(multilabel[1]))
        eva = classification_report(result.truth, result.predictions, labels=multilabel, digits=3)
        roc_auc = "~Not defined for multilabel~"

    print("Evaluation result MV:")
    print(eva)

    ##### save results
    now = datetime.now()  # current date and time
    date_time = now.strftime("%d_%m")
    filename = "./evaluation/baselines_noSTILTS/results/MV_Baseline_" + task + "_" + date_time + ".txt"
    file_exists = os.path.isfile(filename)
    if not file_exists:
        write_evaluationTXT(filename, task,  eva, roc_auc)
    else:
        # ensures that previous results are not overwritten
        raise EvaluationFileAlreadyExists
    print("Done")


def majority_vote_dict(array, label_counts_dict, task):
    print(task, "BASELINE - Majority vote baseline running")

    major_label = toolkit.dict_get_max(label_counts_dict)
    predictions = [major_label] * len(array)
    # for elem in array:
    #     predictions.append(major_label)
    labels = list(label_counts_dict.keys())

    result = Result(label_list=labels, predictions=predictions, truth=array,
                    descr="BL Majority Vote " + task)

    eva = classification_report(result.truth, result.predictions, labels=labels, digits=3)

    #####
    print("Evaluation result MV:")
    print(eva)
    """ ROC AUC"""
    roc_auc = roc_auc_score(y_true=result.truth, y_score=result.predictions, average="macro", sample_weight=None,
                            max_fpr=None)
    print("ROC: " , roc_auc)
    ##### save results
    now = datetime.now()  # current date and time
    date_time = now.strftime("%d_%m")
    filename = "./evaluation/baselines_noSTILTS/results/MV_Baseline_" + task + "_" + date_time + ".txt"
    file_exists = os.path.isfile(filename)
    if not file_exists:
        write_evaluationTXT(filename, task,  eva, roc_auc)
    else:
        # ensures that previous results are not overwritten
        raise EvaluationFileAlreadyExists

    print("Done")


class Result:
    def __init__(self, label_list, predictions, truth, descr):
        self.label_list = label_list
        self.predictions = predictions
        self.truth = truth
        self.description = descr


def write_evaluationTXT(filename, task,  eva, roc):
    now = datetime.now()  # current date and time
    date_time = now.strftime("%d_%m")
    eva_file = open(filename, "a")
    eva_file.write(task + " >> " + date_time)
    eva_file.write("\n")
    eva_file.write(eva)
    eva_file.write("Receiver Operating Characteristic Curve (ROC AUC):")
    eva_file.write("\n")
    eva_file.write("ROC: \t")
    eva_file.write(str(roc))
    eva_file.close()


def main():
    pass
    # corpora = get_InsuffientSupport_datset()
    # train, test = train_test_split(corpora, test_size=0.3, random_state=6)
    # majority_vote(train=train, test=test, column="ANNOTATION", task="Insufficient Supported Arguments")

    # corpora = get_QualityPrediction_dataset()
    # train, test = train_test_split(corpora, test_size=0.3, random_state=6)
    # majority_vote(train=corpora[0], test=corpora[1], column="target_label", task="Argument Quality Prediction")

    # corpora = get_ArgRecognition_dataset(additional_tasks=True)
    # multilabel_ArgRec = [1, 2, 3, 4, 5]
    # majority_vote(corpora[1], corpora[0], "label", "Argument Recogniiton GM", multilabel=multilabel_ArgRec)
    # majority_vote(corpora[0], corpora[1], "label", "Argument Recognition UGIP", multilabel=multilabel_ArgRec)
    # multilabel_ArgRec_3labels = [1, 2, 3]
    # majority_vote(corpora[1], corpora[0], "task_3labels", "Argument Recogniiton GM - 3 Labels", multilabel=multilabel_ArgRec_3labels)
    # majority_vote(corpora[0], corpora[1], "task_3labels", "Argument Recognition UGIP - 3 Labels", multilabel=multilabel_ArgRec_3labels)
    # corpora_whole = corpora[0].append(corpora[1], ignore_index=True)
    # print(corpora_whole.head())
    # print(corpora_whole.describe())
    # print(corpora_whole.groupby("comment_text").head())
    # print(corpora_whole.label.value_counts())
    # majority_vote(corpora_whole, "label", "Argument Recognition  GM & UGIP", multilabel=multilabel_ArgRec)

    # corpora = get_ArgZoning_dataset()
    # multilabel_AZI = corpora.AZ_category.unique()
    # train, test = train_test_split(corpora, test_size=0.3, random_state=6)
    # majority_vote(train=train, test=test, column="AZ_category", task="Argument Zoning I", multilabel=multilabel_AZI)

    ACI_Annotation_Habernal = brat_annotations.parse_annotations_Habernal()
    # #ACI_Annotation_Habernal = ACI_Annotation_Habernal["Label"].apply(pd.DataFrame.to_string)
    # #print(ACI_Annotation_Habernal.head())
    # multilabel_ACI_Habernal = ACI_Annotation_Habernal.target_label.unique()
    # train, test = train_test_split(ACI_Annotation_Habernal, test_size=0.3, random_state=6)
    # majority_vote(train=train, test=test, column="target_label", task="ACI Habernal",
    #               multilabel=multilabel_ACI_Habernal)

    # x, y_arg, y_rhet = load_conll.load_data(
    #     path=r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Lauscher"
    #          r"\annotations_conll_all_splitted\train_dev_Thesis")
    # x_test, y_arg_test, y_rhet_test = load_conll.load_data(
    #     path=r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Lauscher"
    #          r"\annotations_conll_all_splitted\test")
    # labels_lauscher = toolkit.get_unique_values(y_arg)
    # unnested_y_arg = toolkit.unnest_twofold_array(y_arg_test)
    # majority_vote_dict(unnested_y_arg, labels_lauscher, "ACI Lauscher")


if __name__ == '__main__':
    main()
