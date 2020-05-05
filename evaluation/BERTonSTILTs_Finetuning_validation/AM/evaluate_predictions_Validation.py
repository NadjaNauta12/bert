import os
import codecs
import platform
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    classification_report
import sys


from util.PathMux import get_path_to_OS


def _write_eva_results(path, accuracy, f1_score, recall, precision, task, config, setting_AR):
    # path ist ueberdir where models are located
    container_file = "Evaluation_across_" + task + ".tsv"
    if task == "ArgRecognition":
        container_file = "Evaluation_across_" + task + "_" + setting_AR + ".tsv"
    with open(path + "/" + container_file, "a") as myfile:
        myfile.write("\n")
        myfile.write("Setting: \t Size: %s \t Learningrate: %s \t Epochs: %s \n" % config)
        myfile.write('Accuracy:\t %s \n' % accuracy)
        myfile.write('F1 score:\t %s \n' % f1_score)
        myfile.write('Recall:\t %s \n' % recall)
        myfile.write('Precision:\t %s \n' % precision)
        myfile.write("\n")
        myfile.close()


def _process_files(files_model, path, task):
    goldlabel = []
    prediction = []
    for file in files_model:
        if '-goldlabels.tsv' in file:
            # print(path, file)
            with codecs.open(os.path.join(path, file), mode="r", encoding="utf8") as gold_file:
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

        elif '-parsed_test_results.tsv' in file and "ACI" not in task:

            with codecs.open(os.path.join(path, file), mode="r", encoding="utf8") as pred_file:
                next(pred_file, None)

                for line in pred_file:
                    prediction.append(line.split("\t")[1].rstrip('\n'))

        elif '-mapped_test_results.tsv' in file and "ACI" in task:

            with codecs.open(os.path.join(path, file), mode="r", encoding="utf8") as pred_file:
                next(pred_file, None)  # has no header

                for line in pred_file:
                    if line == "\n":
                        # prediction.append('\n')
                        continue
                    else:
                        prediction.append(line.split("\t")[2].rstrip('\n').replace("Token_Label.", ""))
    return goldlabel, prediction


def _evaluation(goldlabel, prediction, dir, subdir, task):
    if "ArgRecognition" in dir:
        task, batch, eta, epochs, setting_AR = dir.split("_", 4)
    elif "ACI" in task:
        task, task2, batch, eta, epochs = dir.split("_", 5)
        task = task + "_" + task2
        setting_AR = ""

    else:
        task, batch, eta, epochs = dir.split("_", 3)
        setting_AR = ""
    # print(task)
    # print("SETTING", setting_AR)
    # make eva
    # from sklearn.metrics import precision_recall_fscore_support as score
    # precision, recall, fscore, support = score(goldlabel, prediction, pos_label='micro')
    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))
    # print('fscore: {}'.format(fscore))
    # print('support: {}'.format(support))
    # print()
    # print()

    # print("MACRO")
    accuracy = accuracy_score(goldlabel, prediction)
    f1 = f1_score(goldlabel, prediction, average="macro")
    # print('F1 score:', f1)
    recall = recall_score(goldlabel, prediction, average="macro")
    precision = precision_score(goldlabel, prediction, average="macro")
    classification = classification_report(goldlabel, prediction, digits=3)
    # print()
    # print("MICRO")
    # f1 = f1_score(goldlabel, prediction, average="micro")
    # print('F1 score:', f1)
    # recall = recall_score(goldlabel, prediction, average="micro")
    # precision = precision_score(goldlabel, prediction, average="micro")
    # classification = classification_report(goldlabel, prediction, digits=3)
    #
    # print('Accuracy:', accuracy)
    # print('F1 score:', f1)
    # print('Recall:', recall)
    # print('Precision:', precision)
    # print('\n classification report:\n')
    # print(classification)
    # from nltk import ConfusionMatrix
    # print(ConfusionMatrix(list(goldlabel), list(prediction)))
    # write into one file -> with config string and task
    #path = get_path_to_OS() + "/evaluation/BERTonSTILTs_Finetuning_valiadation/AM"
    #print("OUTPUT PATH", subdir)
    _write_eva_results(subdir, accuracy, f1, recall, precision, task, config=(batch, eta, epochs),
                       setting_AR=setting_AR)


def run_evaluation(path):
    for subdir, dirs, files in os.walk(path):
        for dir in dirs:
            if dir == "MNLI":
                continue
            path = subdir + "/" + dir
            #print("PATH", path)
            task = dir.split("_")[0]
            if task == "ACI":
                task += "_" + dir.split("_")[1]
            for subdir_model, dirs_model, files_model in os.walk(path):
                goldlabel, prediction = _process_files(files_model, path, task)

                #  predictions and goldLabels should be equally long
                if len(prediction) > 0 and len(prediction) == len(goldlabel):
                    _evaluation(goldlabel, prediction, dir, subdir, task)
                else:
                    print("Size of prediction and parsed examples does not fit")
                    # print("Lengths:", len(prediction), "and", len(goldlabel))


def main():
    task_names = ["ArgRecognition_GM", "ACI_Stab", "ArgZoningI", "ArgQuality", "InsufficientArgSupport",
                  "ArgRecognition_GM_UGIP", "ACI_Lauscher"]
    for task in task_names:
        run_evaluation(get_path_to_OS() + "/models_finetuning_validation/" + task)


if __name__ == "__main__":
    main()
