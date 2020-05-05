import os
import codecs
import platform
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    classification_report
import sys

from util.PathMux import get_path_to_OS


def _consolidate_dev_results(path, scores, task, config, setting_AR):
    container_file = "DEV_Eval_across_" + task + ".txt"
    if task == "ArgRecognition":
        container_file = "DEV_Eval_across_" + task + "_" + setting_AR + ".txt"
    with open(get_path_to_OS() + "/evaluation/SingleTaskSetting/" + container_file, "a") as myfile:
        myfile.write("Setting: \t Size: %s \t Learningrate: %s \t Epochs: %s \n" % config)
        myfile.write(scores)
        myfile.write("\n")
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
                        #_evaluation(goldlabel, prediction, dir, subdir, task)
                        if "ArgRecognition" in dir:
                            task, batch, eta, epochs, setting_AR = dir.split("_", 4)
                        elif "ACI" in task:
                            task, task2, batch, eta, epochs = dir.split("_", 5)
                            task = task + "_" + task2
                            setting_AR = ""
                        else:
                            task, batch, eta, epochs = dir.split("_", 3)
                            setting_AR = ""
                        _consolidate_dev_results(subdir, scores, task,  config=(batch, eta, epochs),
                                                 setting_AR=setting_AR)
                    else:
                        print("eval_results.txt could not be found or read:", path)


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
    counter =0
    for subdir, dirs, files in os.walk(path):
        counter+=1
        for dir in dirs:
            path = subdir + "/" + dir

            task = dir.split("_")[0]
            if task == "ACI":
                task += "_" + dir.split("_")[1]

            for subdir_model, dirs_model, files_model in os.walk(path):
                if "eval" in subdir_model:
                    continue
                scores = _process_file(files_model, path)
                if len(scores) > 0:
                    #_evaluation(goldlabel, prediction, dir, subdir, task)
                    if "ArgRecognition" in dir:
                        task, batch, eta, epochs, setting_AR = dir.split("_", 4)
                    elif "ACI" in task:
                        task, task2, batch, eta, epochs = dir.split("_", 5)
                        task = task + "_" + task2
                        setting_AR = ""
                    else:
                        task, batch, eta, epochs = dir.split("_", 3)
                        setting_AR = ""
                    _consolidate_dev_results(subdir, scores, task,  config=(batch, eta, epochs),
                                             setting_AR=setting_AR)
                else:
                    print("eval_results.txt could not be found or read:", path, "~", files_model)





def main():
    dev_eva_onSTILTs = True
    dev_eva_onSTILTs_filtered = False
    dev_eva_MNLI = False

    ' ####################################################################################'
    if dev_eva_onSTILTs:
        run_DEV_evaluation(get_path_to_OS() + "/models_onSTILTs/models")

    if dev_eva_onSTILTs_filtered:
        task = "ArgRecognition_GM"
        run_DEV_evaluation_on_Task(get_path_to_OS() + "/models_onSTILTs/models", task)

    if dev_eva_MNLI:
        task = "MNLI"
        run_DEV_evaluation(get_path_to_OS() + "/models_finetuning_validation/MNLI")


if __name__ == "__main__":
    main()
