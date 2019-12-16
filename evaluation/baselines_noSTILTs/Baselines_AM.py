import pandas as pd
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

import os.path
import toolkit
from datetime import datetime
from sklearn.metrics import classification_report
from custom_exceptions import EvaluationFileAlreadyExists
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from util.load_datasets import ACI_loader_Habernal, AQ_loader, AZ_loader, ISA_loader, AR_loader, load_conll, \
    load_comp, data_loader
from BERT.run_classifier_multipleParameter import ArgZoningIProcessor as AZIProcessor




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


    if multilabel is None:
        eva = classification_report(result.truth, result.predictions, labels=result.label_list, digits=3)
        # """ ROC AUC"""
        # roc_auc = roc_auc_score(y_true=result.truth, y_score=result.predictions, average="macro", sample_weight=None,
        #                     max_fpr=None)
        # roc_auc = 0.0
        # print("ROC: ", roc_auc)
    else:
        eva = classification_report(result.truth, result.predictions, labels=multilabel, digits=3)
        # roc_auc = "~Not defined for multilabel~"

    print("Evaluation result MV:")
    print(eva)

    ##### save results
    now = datetime.now()  # current date and time
    date_time = now.strftime("%d_%m")
    filename = "C:/Users/Wifo/PycharmProjects/Masterthesis/evaluation/baselines_noSTILTs/results/MV_Baseline_" + task + "_" + date_time + ".txt"
    file_exists = os.path.isfile(filename)
    if not file_exists:
        write_evaluationTXT(filename, task, eva, None)
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
    print("ROC: ", roc_auc)
    ##### save results
    now = datetime.now()  # current date and time
    date_time = now.strftime("%d_%m")
    filename = "C:/Users/Wifo/PycharmProjects/Masterthesis/evaluation/baselines_noSTILTs/results/MV_Baseline_" + task + "_" + date_time + ".txt"
    file_exists = os.path.isfile(filename)
    if not file_exists:
        write_evaluationTXT(filename, task, eva, roc_auc)
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


def write_evaluationTXT(filename, task, eva, roc=None):
    now = datetime.now()  # current date and time
    date_time = now.strftime("%d_%m")
    eva_file = open(filename, "w")
    eva_file.write(task + " >> " + date_time)
    eva_file.write("\n")
    eva_file.write(eva)
    if roc != None:
        eva_file.write("Receiver Operating Characteristic Curve (ROC AUC):")
        eva_file.write("\n")
        eva_file.write("ROC: \t")
        eva_file.write(str(roc))
    eva_file.close()


def main():
    pass

    # Currently there is no ROC


    # Final Version
    # train = ISA_loader.get_InsuffientSupport_datset_byFilter(case=1)
    # print(len(train))
    # dev = ISA_loader.get_InsuffientSupport_datset_byFilter(case=2)
    # print(len(dev))
    # train_dev = pd.concat([train, dev], ignore_index=True)
    # test = ISA_loader.get_InsuffientSupport_datset_byFilter(case=3)
    # print(len(test))
    # majority_vote(train=train_dev, test=test, column="ANNOTATION", task="Insufficient Supported Arguments")

    # Final Version
    # train = AQ_loader.load_ArgQuality_datset(1)
    # dev = AQ_loader.load_ArgQuality_datset(2)
    # test = AQ_loader.load_ArgQuality_datset(3)
    # train_dev = pd.concat((train, dev), axis=0)
    # majority_vote(train=train_dev, test=test, column="label", task="Argument Quality Prediction")

    # Final Version - GM Setting
    # train = AR_loader.get_ArgRecognition_GM_dataset(1)
    # dev = AR_loader.get_ArgRecognition_GM_dataset(2)
    #
    # train_dev = pd.concat((train, dev), axis=0)
    # test = AR_loader.get_ArgRecognition_GM_dataset(3)
    # multilabel_ArgRec = [1, 2, 3, 4, 5]
    # majority_vote(train=train_dev, test=test, column="label", task="Argument Recogniiton GM",
    #               multilabel=multilabel_ArgRec)

    # Final Version - GM_UGIP Setting
    # train, dev, test =  AR_loader.get_ArgRecognition_dataset(case_ID=1)
    # train_dev = pd.concat((train, dev), axis=0)
    # multilabel_ArgRec = [1, 2, 3, 4, 5]
    # majority_vote(train=train_dev, test=test, column="label", task="Argument Recogniiton GM_UGIP Setting",
    #               multilabel=multilabel_ArgRec)


    # Final Version
    # train = AZ_loader.get_ArgZoning_dataset(1)
    # dev = AZ_loader.get_ArgZoning_dataset(2)
    # test = AZ_loader.get_ArgZoning_dataset(3)
    # train_dev = pd.concat((train, dev), axis=0)
    # multilabel_AZI = AZIProcessor.get_labels()
    # majority_vote(train=train_dev, test=test, column="AZ_category", task="Argument Zoning I", multilabel=multilabel_AZI)





    # >>>>>>>>>>>>>> REDO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



    # ACI_Annotation_Habernal = brat_annotations.parse_annotations_Habernal()
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
