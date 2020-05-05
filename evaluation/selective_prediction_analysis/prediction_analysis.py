import os
import codecs
import platform
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    classification_report
import sys


from util.PathMux import get_path_to_OS


def _loadGold(path, task):
    goldlabel = []
    if '-goldlabels.tsv' in path:
        with codecs.open(path, mode="r", encoding="utf8") as gold_file:
            next(gold_file, None)  # skip header
            if "ACI" in task:  # for ACI
                for line in gold_file:
                    if line == "\n":
                        # goldlabel.append('\n')
                        continue
                    else:
                        goldlabel.append(line.split("\t")[2].rstrip('\n').replace("Token_Label.", ""))
            else:
                for line in gold_file:
                    goldlabel.append(line.split("\t")[1].rstrip('\n'))
    return goldlabel


def _load_parsed_results(path, task):
    prediction = []

    if '-parsed_test_results.tsv' in path and "ACI" not in task:

        with codecs.open(path, mode="r", encoding="utf8") as pred_file:
            next(pred_file, None)

            for line in pred_file:
                prediction.append(line.split("\t")[1].rstrip('\n'))

    elif '-mapped_test_results.tsv' in path and "ACI" in task:

        with codecs.open(path, mode="r", encoding="utf8") as pred_file:
            next(pred_file, None)  # has no header

            for line in pred_file:
                if line == "\n":
                    # prediction.append('\n')
                    continue
                else:
                    prediction.append(line.split("\t")[2].rstrip('\n').replace("Token_Label.", ""))
    return prediction


def evaluation(goldlabel, prediction, text):
    print(text)
    print("MACRO")
    accuracy = accuracy_score(goldlabel, prediction)
    f1 = f1_score(goldlabel, prediction, average="macro")
    # print('F1 score:', f1)
    recall = recall_score(goldlabel, prediction, average="macro")
    precision = precision_score(goldlabel, prediction, average="macro")
    classification = classification_report(goldlabel, prediction, digits=3)
    print('Accuracy:', accuracy)
    print('F1 score:', f1)
    print('Recall:', recall)
    print('Precision:', precision)
    print('\n classification report:\n')
    print(classification)
    from nltk import ConfusionMatrix
    print(ConfusionMatrix(list(goldlabel), list(prediction)))


def main():
    # >> '-goldlabels.tsv' files
    goldLabels_path = get_path_to_OS() + \
                      "/models_onSTILTs/models/ArgRecognition_32_2e-05_4_GM_UGIP/ArgRecognition" + \
                      "-goldlabels.tsv"
    # >> '-parsed_test_results.tsv' Or for ACI '-mapped_test_results.tsv'
    STS_path = get_path_to_OS() + \
               "/models_onSTILTs/models/ArgRecognition_32_2e-05_4_GM_UGIP/ArgRecognition" \
               + "-parsed_test_results.tsv"

    # >> '-parsed_test_results.tsv' Or for ACI '-mapped_test_results.tsv'
    BERTonSTILT_path = get_path_to_OS() + \
                       "/models_finetuning/ArgZoningI/ArgRecognition_32_3e-05_4_GM_UGIP/ArgRecognition" \
                       + "-parsed_test_results.tsv"

    BERTonSTILT_path2 = get_path_to_OS() + \
                        "/models_finetuning/ACI_Lauscher/ArgRecognition_32_3e-05_4_GM_UGIP/ArgRecognition" \
                        + "-parsed_test_results.tsv"

    task = "AR_2"

    task_set = {
        "AQ": "ArgumentQuality",
        "ISA": "InsufficientArgSupport",
        "AR_1": "ArgumentRecognition_GM",
        "AR_2": "ArgumentRecognition_GM_UGIP",
        "ACI_L": "ArgumentComponentIdentification_Lauscher",
        "ACI_S": "ArgumentComponentIdentification_Stab",
        "AZ": "ArgumentZonningI"
    }

    eva1 = False
    eva2 = False
    eva3 = False
    eva4 = False
    eva5 = True
    if eva1:
        task = "AR_2"
        if task not in task_set:
            raise ValueError("Task not found: %s" % task)
        evaluation(_loadGold(goldLabels_path, task), _load_parsed_results(STS_path, task), "GOLD-STS_EVALUATION")
        evaluation(_loadGold(goldLabels_path, task), _load_parsed_results(BERTonSTILT_path, task),
                   "GOLD-STILT AZ_EVALUATION")
        valuation(_loadGold(goldLabels_path, task), _load_parsed_results(BERTonSTILT_path2, task),
                  "GOLD-STILT ACIL_EVALUATION")

    if eva2:
        task = "AR_1"
        if task not in task_set:
            raise ValueError("Task not found: %s" % task)
        # >> '-goldlabels.tsv' files
        goldLabels_path = get_path_to_OS() + \
                          "/models_onSTILTs/models/ArgRecognition_32_2e-05_3_GM/ArgRecognition" + \
                          "-goldlabels.tsv"
        # >> '-parsed_test_results.tsv' Or for ACI '-mapped_test_results.tsv'
        STS_path = get_path_to_OS() + \
                   "/models_onSTILTs/models/ArgRecognition_32_2e-05_3_GM/ArgRecognition" \
                   + "-parsed_test_results.tsv"

        # >> '-parsed_test_results.tsv' Or for ACI '-mapped_test_results.tsv'
        BERTonSTILT_path = get_path_to_OS() + \
                           "/models_finetuning/ACI_Stab/ArgRecognition_32_3e-05_4_GM/ArgRecognition" \
                           + "-parsed_test_results.tsv"

        BERTonSTILT_path2 = get_path_to_OS() + \
                            "/models_finetuning/ACI_Lauscher/ArgRecognition_32_3e-05_4_GM_UGIP/ArgRecognition" \
                            + "-parsed_test_results.tsv"

        evaluation(_loadGold(goldLabels_path, task), _load_parsed_results(STS_path, task), "GOLD-STS_EVALUATION")
        evaluation(_loadGold(goldLabels_path, task), _load_parsed_results(BERTonSTILT_path, task),
                   "GOLD-STILT AZ_EVALUATION")
        # evaluation(_loadGold(goldLabels_path, task), _load_parsed_results(BERTonSTILT_path2, task), "GOLD-STILT ACIL_EVALUATION")

    if eva3:
        task = "AZ"
        if task not in task_set:
            raise ValueError("Task not found: %s" % task)
        # >> '-goldlabels.tsv' files
        goldLabels_path = get_path_to_OS() + \
                          "/models_onSTILTs/models/ArgZoningI_32_2e-05_3/ArgZoningI" + \
                          "-goldlabels.tsv"
        # >> '-parsed_test_results.tsv' Or for ACI '-mapped_test_results.tsv'
        # STS_path = get_path_to_OS() + \
        #            "/models_onSTILTs/models/ArgRecognition_32_2e-05_3_GM/ArgRecognition" \
        #            + "-parsed_test_results.tsv"

        # >> '-parsed_test_results.tsv' Or for ACI '-mapped_test_results.tsv'
        BERTonSTILT_path = get_path_to_OS() + \
                           "/models_finetuning/ACI_Stab/ArgZoningI_32_2e-05_3/ArgZoningI" \
                           + "-parsed_test_results.tsv"

        BERTonSTILT_path2 = get_path_to_OS() + \
                            "/models_finetuning/ACI_Lauscher/ArgZoningI_32_3e-05_3/ArgZoningI" \
                            + "-parsed_test_results.tsv"

        evaluation(_loadGold(goldLabels_path, task), _load_parsed_results(BERTonSTILT_path, task),
                   "ACIS >> AZ EVALUATION")
        evaluation(_loadGold(goldLabels_path, task), _load_parsed_results(BERTonSTILT_path2, task),
                   "ACIL >> AZ EVALUATION")
    if eva4:
        task = "ACI_L"
        if task not in task_set:
            raise ValueError("Task not found: %s" % task)
        # >> '-goldlabels.tsv' files
        goldLabels_path = get_path_to_OS() + \
                          "/models_onSTILTs/models/ACI_Lauscher_32_2e-05_4/ACI_Lauscher" + \
                          "-goldlabels.tsv"
        # >> '-parsed_test_results.tsv' Or for ACI '-mapped_test_results.tsv'
        STS_path = get_path_to_OS() + \
                   "/models_onSTILTs/models/ACI_Lauscher_32_2e-05_4/ACI_Lauscher" \
                   + "-mapped_test_results.tsv"

        # >> '-parsed_test_results.tsv' Or for ACI '-mapped_test_results.tsv'
        BERTonSTILT_path = get_path_to_OS() + \
                           "/models_finetuning/ACI_Stab/ACI_Lauscher_32_3e-05_4/ACI_Lauscher" \
                           + "-mapped_test_results.tsv"

        #evaluation(_loadGold(goldLabels_path, task), _load_parsed_results(STS_path, task), "GOLD-STS_EVALUATION")
        evaluation(_loadGold(goldLabels_path, task), _load_parsed_results(BERTonSTILT_path, task),
                   "GOLD-STILT ACI Stab_EVALUATION")

    if eva5:
        task = "ACI_S"
        if task not in task_set:
            raise ValueError("Task not found: %s" % task)
        # >> '-goldlabels.tsv' files
        goldLabels_path = get_path_to_OS() + \
                          "/models_onSTILTs/models/ACI_Stab_32_2e-05_3/ACI_Stab" + \
                          "-goldlabels.tsv"
        # >> '-parsed_test_results.tsv' Or for ACI '-mapped_test_results.tsv'
        STS_path = get_path_to_OS() + \
                   "/models_onSTILTs/models/ACI_Stab_32_2e-05_3/ACI_Stab" \
                   + "-mapped_test_results.tsv"

        # >> '-parsed_test_results.tsv' Or for ACI '-mapped_test_results.tsv'
        BERTonSTILT_path = get_path_to_OS() + \
                           "/models_finetuning/ACI_Lauscher/ACI_Stab_32_2e-05_4/ACI_Stab" \
                           + "-mapped_test_results.tsv"

        #evaluation(_loadGold(goldLabels_path, task), _load_parsed_results(STS_path, task), "GOLD-STS_EVALUATION")
        evaluation(_loadGold(goldLabels_path, task), _load_parsed_results(BERTonSTILT_path, task),
                   "GOLD-STILT ACI Lauscher_EVALUATION")


if __name__ == "__main__":
    main()
