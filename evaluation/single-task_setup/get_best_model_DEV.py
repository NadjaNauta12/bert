import os
import codecs
import sys

from util.PathMux import get_path_to_OS

# for each task
# open file
# read every line and compare to best model
#    if better remember model and value


def _write_best_model_overview(path, list_of_models, filename=None):
    path_output = path + "/Best_DEV_Score.txt"
    if filename is not None:
        path_output = path + "/" + filename

    with codecs.open(path_output, "w", "utf-8") as f_out:
        for ele in list_of_models:
            task = ele[0]
            best_setting = ele[1]
            measures = ele [2]
            f_out.write("Best model for\t" +  task +"\n")
            f_out.write("Setting\t" + best_setting+ "\n")
            f_out.write("Performance:\n")
            for key, val in measures.items():
                f_out.write(key + "\t" + val + "\n")
            f_out.write("\n")
        f_out.close()


def get_best_model(path, task):
    best_setting = ""
    best_performance = {'eval_accuracy': 0.0, 'eval_loss': 100.0, 'loss': 100.0, "global_step": -1}
    measure_dict = {}
    cur_setting = ""
    for subdir, dirs, files in os.walk(path):
        for file in files:

            if task in file and "DEV_Evaluation_across" in file:


                if task == "ArgRecognition_GM" in file and "GM_UGIP" in file: # since settings share prefix
                    continue


                with codecs.open(os.path.join(path, file), mode="r", encoding="utf8") as dev_scores_file:
                    for line in dev_scores_file:
                        if line == "\n" or line == "\r\n":
                            if len(measure_dict) > 0:
                                if float(measure_dict.get('eval_accuracy', 0.0)) \
                                        > float(best_performance.get('eval_accuracy')):
                                    best_setting = cur_setting
                                    for key in measure_dict.keys():
                                        best_performance[key] = measure_dict.get(key)
                                    measure_dict = {}

                                # better_model = False
                                # for key in best_performance.keys():
                                #     if measure_dict.get(key, default= 0.0) > best_performance.get(key):
                                #          better_model= True
                                # if better_model:
                                #     best_setting = cur_setting
                            pass
                        elif "Setting" in line:
                            cur_setting = line.rstrip("\r\n")
                        else:
                            measure, value = line.strip("\r\n").split("=")
                            measure_dict[measure.strip()] = value.strip()


    print("Best model for ", task)
    print("Setting", best_setting)
    print("Performance:")
    for key, val in best_performance.items():
        print(key, val)
    return (task, best_setting, best_performance)


def _get_best_dev_score_SingleTaskSetting():
    best_models = []
    for task in ["ACI_Stab", "ArgRecognition_GM_UGIP", "ArgZoningI", "ArgQuality", "ArgRecognition_GM",
                 "InsufficientArgSupport", "ACI_Lauscher"]:
    
        path = get_path_to_OS() + "/evaluation/SingleTaskSetting"
        best_model = get_best_model(path, task)
        if best_model[1] is not "":
            best_models.append(best_model)

    _write_best_model_overview(get_path_to_OS() + "/evaluation/SingleTaskSetting", best_models)


def _get_best_dev_score_MNLI():
    best_models = []
    for task in ["MNLI"]:

        path = get_path_to_OS() + "/evaluation/SingleTaskSetting"
        best_model = get_best_model(path, task)
        if best_model[1] is not "":
            best_models.append(best_model)

    _write_best_model_overview(get_path_to_OS() + "/evaluation/SingleTaskSetting", best_models,
                               filename="Best_MNLI_DEV_Score.txt")


def main():
    # _get_best_dev_score_SingleTaskSetting()

    _get_best_dev_score_MNLI()


if __name__ == "__main__":
    main()
