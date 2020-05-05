import os
import codecs
import sys


from util.PathMux import get_path_to_OS


def _write_dev_score_overview(path, list_of_models, task_combination):
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


def get_all_dev_scores_for_task(path, task):
    settings = []
    dev_scores = []
    measure_dict = {}
    cur_setting = ""
    for subdir, dirs, files in os.walk(path):
        for model_dir in dirs:
            if task in model_dir:
                if task == "ArgRecognition_GM" and "_GM_UGIP" in file:
                    continue

                for file in os.listdir(os.path.join(path, model_dir)):
                    if "eval_results.txt" == file:
                        with codecs.open(os.path.join(path,model_dir, file), mode="r", encoding="utf8") as dev_file:
                            scores = dev_file.read()
                            dev_scores.append(scores)
                            if "ArgRecognition" in model_dir:
                                task, batch, eta, epochs, setting_AR = model_dir.split("_", 4)
                            elif "ACI" in model_dir:
                                task, task2, batch, eta, epochs = model_dir.split("_", 5)
                                task = task + "_" + task2
                            else:
                                task, batch, eta, epochs = model_dir.split("_", 3)

                            settings.append((batch, eta, epochs))
                        dev_file.close()
                        break

    assert (len(dev_scores) == len(settings))
    return (dev_scores, settings)

def main():
    task_names = ["ACI_Lauscher", "ACI_Stab", "ArgZoningI", "ArgQuality", "InsufficientArgSupport",
                  "ArgRecognition_GM_UGIP", "ArgRecognition"]

    FineTuning = False
    if FineTuning:
        for i, name in enumerate(task_names):

            all_dev_scores_dict = {}  # list of model tuples

            for ii, target in enumerate(task_names):
                if i == ii:
                    continue
                combination=name + "/" + target
                path = get_path_to_OS() + "/models_finetuning/" + name

                dev_scores, settings = get_all_dev_scores_for_task(path, target)  # tuple of 4
                if len(dev_scores) >0 and len(settings) >0:
                    all_dev_scores_dict[combination] = (dev_scores, settings)

            print(all_dev_scores_dict.keys())

    MNLI_Validation = True
    if MNLI_Validation:
        all_dev_scores_dict = []  # list of model tuples
        combinations = []
        for i, name in enumerate(task_names):
            combinations.append("MNLI/" + name)
            path = get_path_to_OS() + "/models_finetuning_validation/MNLI/"
            models = get_all_dev_scores_for_task(path, name)  # tuple of 4
            all_dev_scores_dict.append(models)

        print(len(models))  # == 4  # tuple of 4
        print(len(combinations))  # == 6  # 6 other tasks targeted -string
        print(len(all_dev_scores_dict))  # == 6  # 6 other tasks targeted -models

        assert len(models) == 4  # tuple of 4
        assert len(combinations) == 7  # 6 other tasks targeted -string
        assert len(all_dev_scores_dict) == 7  # 6 other tasks targeted -models

        _write_dev_score_overview(get_path_to_OS()
                                  + "/models_finetuning_validation/Consolidated_MNLI_Modelevaluation.txt",
                                  all_dev_scores_dict, combinations)

    Validation = False
    if Validation:
        validation_target = ["ArgZoningI", "ArgQuality", "InsufficientArgSupport"]

        for i, name in enumerate(task_names):

            all_dev_scores_dict = []  # list of model tuples
            combinations = []
            for ii, target in enumerate(validation_target):
                if name == target:
                    continue
                combinations.append(name + "/" + target)
                path = get_path_to_OS() + "/models_finetuning/" + name  # + "/" + target
                models = get_all_dev_scores_for_task(path, target)  # tuple of 4
                all_dev_scores_dict.append(models)

            print("LENGTH", len(combinations), len(all_dev_scores_dict))
            assert len(models) == 4  # tuple of 4
            assert len(combinations) == 3 or len(combinations) == 2  # 3 other tasks targeted -string
            assert len(all_dev_scores_dict) == 3 or len(all_dev_scores_dict) == 2  # 3 other tasks targeted -models

            _write_dev_score_overview(get_path_to_OS()
                                      + "/models_finetuning_validation/Consolidated_Modelevaluation_Validation.txt",
                                      all_dev_scores_dict, combinations)
            all_dev_scores_dict = []  # list of model tuples
            combinations = []


if __name__ == "__main__":
    main()
