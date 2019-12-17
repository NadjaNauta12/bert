import os
import codecs


# for each task
# open file
# read every line and compare to best model
#    if better remember model and value


def get_best_model(path, task):
    best_setting = ""
    best_performance = {'Accuracy:': 0.0, 'F1 score:': 0.0, 'Recall:': 0.0, 'Precision:': 0.0}
    measure_dict = {}
    cur_setting = ""
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if task in file and "Evaluation_across" in file:
                with codecs.open(os.path.join(path, file), mode="r", encoding="utf8") as eva_file:
                    for line in eva_file:
                        if line == "\n" or line == "\r\n":
                            if len(measure_dict) > 0:
                                if float(measure_dict.get('Accuracy:', 0.0)) > float(best_performance.get('Accuracy:')):  # TODO which measure
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
    print("Performance", best_performance)


def main():
    for task in ["ArgRecognition"]: #["ArgZoningI", "ArgQuality", "ArgRecognition", "InsufficientArgSupport"]:
        if os.name == "nt":
            path = "C:/Users/Wifo/PycharmProjects/Masterthesis/models_onSTILTs/models"
        elif platform.release() != "4.9.0-11-amd64":  # GOOGLE COLAB
            print("AQ_Google Colab")
            path = "/content/drive/My Drive/Masterthesis/models_onSTILTs/models"
        else:
            path = "/work/nseemann/models_onSTILTs/models"

        get_best_model(path, task)


if __name__ == "__main__":
    main()
