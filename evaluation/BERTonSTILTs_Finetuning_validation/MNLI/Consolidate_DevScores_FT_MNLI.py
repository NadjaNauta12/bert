import os
import codecs
import platform
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    classification_report
import sys


from util.PathMux import get_path_to_OS


def _consolidate_dev_results(dict_scores):
    container_file = "MNLI_DEV_Scores_Consolidated.txt"

    with open(get_path_to_OS() + "/evaluation/BERTonSTILTs_Finetuning/" + container_file, "w") as myfile:
        for key in dict_scores.keys():
            myfile.write("#####Task Combination\t" + key + "\n")
            # iterate over all dev scores for each model - same combination tasks -hyperparameters different
            for ele in dict_scores.get(key):
                scores, batch, eta, epochs = ele
                myfile.write("Setting: \t Size: %s \t Learningrate: %s \t Epochs: %s \n" % (batch, eta, epochs))
                myfile.write(scores)
                myfile.write("\n")
            myfile.write("#############################################################################################"
                         + "\n")
    myfile.close()


def run_DEV_evaluation_on_Task(path, task):
    for subdir, dirs, files in os.walk(path):
        for dir in dirs:
            if dir == "DEV":
                continue
            if task in dir:
                task = dir.split("_")[0]
                if task == "ACI":
                    task += "_" + dir.split("_")[1]

                path = path + "/" + dir
                for subdir_model, dirs_model, files_model in os.walk(path):

                    scores = _process_file(files_model, path)

                    if len(scores) > 0:
                        # _evaluation(goldlabel, prediction, dir, subdir, task)
                        if "ArgRecognition" in dir:
                            task, batch, eta, epochs, setting_AR = dir.split("_", 4)
                        elif "ACI" in task:
                            task, task2, batch, eta, epochs = dir.split("_", 5)
                            task = task + "_" + task2
                            setting_AR = ""
                        else:
                            task, batch, eta, epochs = dir.split("_", 3)
                            setting_AR = ""
                        _consolidate_dev_results(subdir, scores, task, config=(batch, eta, epochs),
                                                 setting_AR=setting_AR)
                    else:
                        print("eval_results.txt could not be found or read", path)


def _process_file(files_model, path):
    content = None
    for file in files_model:
        if 'eval_results.txt' in file:
            with codecs.open(os.path.join(path, file), mode="r", encoding="utf8") as dev_file:
                content = dev_file.read()
            dev_file.close()
            break
    # no eval_results.txt found in dir ( case for dir /eval)
    if content is None:
        return ""
    return content


def run_DEV_evaluation(path):
    dict_scores = {}
    for trained_model in os.listdir(path):
        subdir_lvl1 = os.path.join(path, trained_model)
        if os.path.isdir(subdir_lvl1):  # trainde models
            scores = _process_file(os.listdir(subdir_lvl1), subdir_lvl1)  # get scores from eval_results.txt

            if len(scores) > 0:
                if "ArgRecognition" in trained_model:
                    task, batch, eta, epochs, setting_AR = trained_model.split("_", 4)
                elif "ACI" in trained_model:
                    task, task2, batch, eta, epochs = trained_model.split("_", 5)
                    task = task + "_" + task2
                    setting_AR = ""
                else:
                    task, batch, eta, epochs = trained_model.split("_", 3)
                    setting_AR = ""
                # print("STILT:", directory, task, batch, eta, epochs, setting_AR)
                target_model = "MNLI>>" + task
                if "ArgRecognition" in trained_model:
                    target_model = target_model + "_" + setting_AR

                bucket = dict_scores.get(target_model)
                if bucket is None:
                    dict_scores[target_model] = [(scores, batch, eta, epochs)]
                else:
                    bucket.append((scores, batch, eta, epochs))

            else:
                print("eval_results.txt could not be found or read", path)
    _consolidate_dev_results(dict_scores)


def main():
    run_DEV_evaluation(get_path_to_OS() + "/models_finetuning_validation/MNLI")


if __name__ == "__main__":
    main()
