import os
import codecs
import sys

from util.PathMux import get_path_to_OS

# for each task
# open file
# read every line and compare to best model
#    if better remember model and value


def _write_best_model_overview(path, list_of_models):
    path_output = path + "/BestModels.txt"
    with codecs.open(path_output, "w", "utf-8") as f_out:
        for ele in list_of_models:
            # ele content: (task, best_setting, measure_dict)
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
    best_performance = {'Accuracy:': 0.0, 'F1 score:': 0.0, 'Recall:': 0.0, 'Precision:': 0.0}
    measure_dict = {}
    cur_setting = ""
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if task in file and "Evaluation_across" in file: # todo check for AR_GM
                with codecs.open(os.path.join(path, file), mode="r", encoding="utf8") as eva_file:
                    for line in eva_file:
                        if line == "\n" or line == "\r\n":
                            if len(measure_dict) > 0:
                                if float(measure_dict.get('F1 score:', 0.0)) > float(best_performance.get('F1 score:')):
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
                            measure, value = line.strip("\r\n").split("\t")
                            measure_dict[measure] = value
    print("Best model for ", task)
    print("Setting", best_setting)
    print("Performance:")
    for key, val in best_performance.items():
        print(key, val)
    return (task, best_setting, best_performance)


def _get_best_model_onSTILTs():
    best_models = []
    #for task in ["ACI_Stab","ArgZoningI", "ArgQuality", "ArgRecognition_GM", "InsufficientArgSupport", "ArgRecognition_GM_UGIP"]:
    for task in [ "ArgRecognition_GM",                      "ArgRecognition_GM_UGIP"]:
        path = get_path_to_OS() + "/models_onSTILTs/models"
        best_model = get_best_model(path, task)
        best_models.append(best_model)

    _write_best_model_overview(get_path_to_OS() + "/models_onSTILTs/models", best_models)


# def _get
#
# def _get_best_model_finetuning():
#     for task in ["ACI_Stab","ArgZoningI", "ArgQuality", "ArgRecognition_GM", "InsufficientArgSupport", "ArgRecognition_GM_UGIP"]:
#         path = get_path_to_OS() + "/models_fine/models"
#         get_best_model(path, task)

def main():
    _get_best_model_onSTILTs()

if __name__ == "__main__":
    main()
