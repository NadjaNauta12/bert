import os
import codecs
import sys


from util.PathMux import get_path_to_OS


def _write_best_model_overview(path, list_of_models, task_combination):
    with codecs.open(path, "a", "utf-8") as f_out:
        for i, model_tuple in enumerate(list_of_models):
            f_out.write("#####Task Combination\t" + task_combination[i] + "\n")

            task = model_tuple[0]
            # f_out.write("Target model :\t" +  task +"\n")
            setting = model_tuple[1]
            measures = model_tuple[2]
            assert len(measures) == len(setting)
            for i, single_model in enumerate(measures):
                f_out.write("Target model :\t" + task + "\n")
                f_out.write(setting[i] + "\n")
                f_out.write("Performance:\n")
                for key, val in single_model.items():
                    f_out.write(key + "\t" + val + "\n")
                f_out.write("\n")
            f_out.write("#############################################################################################"
                        + "\n")
        f_out.close()


def get_all_models_for_task(path, task):
    settings = []
    models = []
    measure_dict = {}
    cur_setting = ""
    for subdir, dirs, files in os.walk(path):
        for file in files:
            # file_names not distrinct for AR...
            if task == "ArgRecognition_GM" and "_GM_UGIP" in file:
                continue
            if task in file and "Evaluation_across" in file:
                with codecs.open(os.path.join(path, file), mode="r", encoding="utf8") as eva_file:
                    for line in eva_file:
                        if line == "\n" or line == "\r\n":
                            if len(measure_dict) > 0:
                                models.append(measure_dict)
                                settings.append(cur_setting)
                                measure_dict = {}
                            pass
                        elif "Setting" in line:
                            cur_setting = line.rstrip("\r\n")
                        else:
                            measure, value = line.strip("\r\n").split("\t")
                            measure_dict[measure] = value
    # print("LEn", task, len(settings), len(models))
    return task, settings, models, path


def main():
    task_names = ["ArgRecognition_GM", "ACI_Stab", "ArgZoningI", "ArgQuality", "InsufficientArgSupport",
                  "ArgRecognition_GM_UGIP", "ACI_Lauscher"]
    for i, name in enumerate(task_names):

        all_target_models = []  # list of model tuples
        combinations = []
        for ii, target in enumerate(task_names):
            if i == ii:
                continue
            combinations.append(name + "/" + target)
            path = get_path_to_OS() + "/models_finetuning_validation/" + name  # + "/" + target
            models = get_all_models_for_task(path, target)  # tuple of 4
            all_target_models.append(models)

        assert len(models) == 4  # tuple of 4
        assert len(combinations) == 6  # 6 other tasks targeted -string
        assert len(all_target_models) == 6  # 6 other tasks targeted -models

        _write_best_model_overview(get_path_to_OS() +
                                   "/evaluation/BERTonSTILTs_Finetuning_validation/AM/Consolidated_Modelevaluation.txt",
                                   all_target_models, combinations)



if __name__ == "__main__":
    main()
