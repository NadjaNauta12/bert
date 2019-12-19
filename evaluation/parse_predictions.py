import sys

sys.path.append("C:/Users/Wifo/PycharmProjects/Masterthesis")
sys.path.append("/work/nseemann")
sys.path.append("/content/bert")
# print(sys.path)
import codecs
import itertools
import numpy as np
import os
import argparse
from run_classifier_multipleParameter import ArgRecognitionProcessor, set_experimental_setting_ARTask


def parse_predictions(input_path, output_path, task):
    """
    Adapted from A. Lauscher
    :param input_path:
    :param output_path:
    :param task:
    :return:

    """

    predicted_labels = []
    if task == "ArgQuality":
        labels = ["a1", "a2"]
        # run_classifier_multipleParameter.ArgQualityProcessor().get_labels()
    elif task == "ArgRecognition":
        labels = [1, 2, 3, 4, 5]
        # run_classifier_multipleParameter.ArgRecognitionProcessor().get_labels()
    elif task == "ArgZoningI":
        labels = ['BKG', 'OTH', 'CTR', 'AIM', 'BAS', 'OWN', 'TXT']
        #  run_classifier_multipleParameter.ArgZoningIProcessor().get_labels()
    elif task == "InsufficientArgSupport":
        labels = ["sufficient", "insufficient"]
        #  run_classifier_multipleParameter.InsufficientSupportProcessor().get_labels()
    else:
        return

    with codecs.open(input_path, "r", "utf8") as f_in:
        for line in f_in.readlines():
            predictions = np.array(line.split("\t"), dtype=np.float32)

            predicted_index = np.argmax(predictions)
            # if predicted_index != 2:
            #     print("Not like the others")
            predicted_labels.append(labels[predicted_index])
            # else:
            #     predicted_labels.append(predictions[0])
        f_in.close()

    with codecs.open(output_path, "w", "utf8") as f_out:
        f_out.write("index\tprediction\n")
        for i, prediction in enumerate(predicted_labels):
            f_out.write(str(i) + "\t" + str(prediction) + "\n")
        f_out.close()


def main():
    parser = argparse.ArgumentParser(description="Running prediction parser")
    parser.add_argument("--task", type=str, default=None,
                        help="Task for which files should be parsed", required=False)
    parser.add_argument("--input_path", type=str,
                        default="/work/nseemann/models_onSTILTs/models",
                        help="Input path of model folder", required=False)
    # parser.add_argument("--output_path_root", type=str,
    #                     default="/work/nseemann/models_onSTILTs",
    #                     help="Input path in case train and dev are in a single file", required=False)
    parser.add_argument("--train_batch_size", type=str,
                        default="[32]",
                        help="Config - Batch", required=False)
    parser.add_argument("--learning_rate", type=str,
                        default="[5e-5, 3e-5, 2e-5]",
                        help="Config - eta", required=False)
    parser.add_argument("--num_train_epochs", type=str,
                        default="[3, 4]",
                        help="Config- Epochs", required=False)

    args = parser.parse_args()
    learning_rate_list = eval(args.learning_rate)
    train_epochs_list = eval(args.num_train_epochs)
    train_batch_sizes_list = eval(args.train_batch_size)
    learning_rate_list = [2e-05]
    train_epochs_list = [2]
    train_batch_sizes_list = [32]

    configs = list(itertools.product(train_batch_sizes_list, learning_rate_list, train_epochs_list))

    for (train_batch_size, learning_rate, train_epochs) in configs:
        config_str = str(train_batch_size) + "_" + str(learning_rate) + "_" + str(train_epochs)
        task = args.task
        task = "ArgRecognition"
        print(args.input_path)
        if task == "ArgRecognition":
            set_experimental_setting_ARTask()
            setting_AR = "_" + ArgRecognitionProcessor.get_experimental_setting()
            input_path = args.input_path + "_" + config_str + setting_AR + "/test_results.tsv"
            output_path = args.input_path + "_" + config_str + setting_AR + "/" + str(task) + "-parsed_test_results.tsv"
        else:
            input_path = args.input_path + "_" + config_str + "/test_results.tsv"
            output_path = args.input_path + "_" + config_str + "/" + str(task) + "-parsed_test_results.tsv"

        a = "C:/Users/Wifo/PycharmProjects/Masterthesis/models_onSTILTs/models"
        input_path = a + "/" + str(task) + "_" + config_str + setting_AR+ "/test_results.tsv"
        output_path = a + "/" + str(task) + "_" + config_str + setting_AR+ "/" + str(task) + "-parsed_test_results.tsv"
        # print(input_path)
        # print(output_path)
        parse_predictions(input_path, output_path, task=task)


if __name__ == "__main__":
    main()
