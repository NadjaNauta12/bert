import os
import codecs
import platform
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    classification_report
import sys


from util.PathMux import get_path_to_OS


def _consolidate_dev_results(dict_scores):
    container_file = "DEV_Scores_Consolidated.txt"

    with open(get_path_to_OS() + "/evaluation/BERTonSTILTs_Finetuning_validation/AM/" + container_file, "w") as myfile:
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
    for directory in os.listdir(path):
        if "MNLI" in directory:
            continue
        subdir_lvl1 = os.path.join(path, directory)
        if os.path.isdir(subdir_lvl1):  # walk thorugh superior filder which BERT aa base
            for trained_model in os.listdir(subdir_lvl1):
                subdir_lvl2 = os.path.join(subdir_lvl1, trained_model)
                if os.path.isdir(subdir_lvl2):  # model folder - finetuned model
                    scores = _process_file(os.listdir(subdir_lvl2), subdir_lvl2)  # get scores from eval_results.txt

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
                        target_model = directory + ">>" + task
                        if "ArgRecognition" in trained_model:
                            target_model = target_model + "_" + setting_AR
                        # print(target_model)

                        bucket = dict_scores.get(target_model)
                        if bucket is None:
                            dict_scores[target_model] = [(scores, batch, eta, epochs)]
                        else:
                            bucket.append((scores, batch, eta, epochs))

                    else:
                        print("eval_results.txt could not be found or read", path)
    _consolidate_dev_results(dict_scores)


def main():
    run_DEV_evaluation(get_path_to_OS() + "/models_finetuning_validation/")


if __name__ == "__main__":
    main()
