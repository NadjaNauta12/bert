import sys

import codecs
import itertools
import numpy as np
import os
import argparse
from run_classifier_multipleParameter import ArgRecognitionProcessor, set_experimental_setting_ARTask
from evaluation.ACI_map_evaluation import parse_ACI_results


def parse_predictions(input_path, output_path, task):
    """
    Adapted from A. Lauscher
    :param input_path:
    :param output_path:
    :param task:
    :return:

    """

    if task == "ArgQuality":
        labels = ["a1", "a2"]
        # run_classifier_multipleParameter.ArgQualityProcessor().get_labels()
    elif task == "ArgRecognition":
        labels = [1, 2, 3, 4, 5]
        # run_classifier_multipleParameter.ArgRecognitionProcessor().get_labels()
    elif task == "ArgZoningI":
        labels = ['BKG', 'OTH', 'CTR', 'AIM', 'BAS', 'OWN', 'TXT', ]
        #  run_classifier_multipleParameter.ArgZoningIProcessor().get_labels()
    elif task == "InsufficientArgSupport":
        labels = ["sufficient", "insufficient"]
        #  run_classifier_multipleParameter.InsufficientSupportProcessor().get_labels()
    elif task == "ACI_Stab":
        labels = ['Token_Label.BEGIN_MAJOR_CLAIM', 'Token_Label.INSIDE_MAJOR_CLAIM', 'Token_Label.BEGIN_CLAIM',
                  'Token_Label.INSIDE_CLAIM', 'Token_Label.BEGIN_PREMISE', 'Token_Label.INSIDE_PREMISE',
                  'Token_Label.OUTSIDE', '[CLS]', '[SEP]']
    elif task == "ACI_Lauscher":
        labels = ['Token_Label.OUTSIDE', 'Token_Label.BEGIN_BACKGROUND_CLAIM', 'Token_Label.INSIDE_BACKGROUND_CLAIM',
                  'Token_Label.BEGIN_DATA', 'Token_Label.INSIDE_DATA', 'Token_Label.BEGIN_OWN_CLAIM',
                  'Token_Label.INSIDE_OWN_CLAIM', '[CLS]', '[SEP]']
    elif task == "MNLI":
        return ["contradiction", "entailment", "neutral"]
    else:
        return

    predicted_labels = []
    with codecs.open(input_path, "r", "utf8") as f_in:
        for line in f_in.readlines():
            if line == "\n":  # req. for ACI Tasks due to sequence
                predicted_labels.append("SEPARATOR")
            else:
                predictions = np.array(line.split("\t"), dtype=np.float32)
                predicted_index = np.argmax(predictions)
                predicted_labels.append(labels[predicted_index])

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
                        default="/models_onSTILTs/models",
                        help="Input path of model folder", required=False)
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

    configs = list(itertools.product(train_batch_sizes_list, learning_rate_list, train_epochs_list))

    for (train_batch_size, learning_rate, train_epochs) in configs:
        config_str = str(train_batch_size) + "_" + str(learning_rate) + "_" + str(train_epochs)
        task = args.task

        if task == "ArgRecognition":
            idx =[0, 1]
            # consider path is slightly different for finetuning
            if "_GM_UGIP" in args.input_path:
                idx.remove(0)
            elif "_GM" in args.input_path:
                idx.remove(1)

            for i in idx:
                if i == 0:
                    # Parse GM Setting
                    set_experimental_setting_ARTask("GM")
                else:
                    # Parse GM->UGIP Setting
                    set_experimental_setting_ARTask("GM_UGIP")

                setting_AR = "_" + ArgRecognitionProcessor.get_experimental_setting()
                input_path = args.input_path + "_" + config_str + setting_AR + "/test_results.tsv"
                # consider path is slightly different for finetuning
                output_path = args.input_path + "_" + config_str + setting_AR + "/" + str(task) + "-parsed_test_results.tsv"
                parse_predictions(input_path, output_path, task=task)


        else:
            input_path = args.input_path + "_" + config_str + "/test_results.tsv"
            output_path = args.input_path + "_" + config_str + "/" + str(task) + "-parsed_test_results.tsv"

            parse_predictions(input_path, output_path, task=task)

            if "ACI" in task:
                # print(args.input_path + "_" + config_str)
                parse_ACI_results(args.input_path + "_" + config_str, task, False)


if __name__ == "__main__":
    main()
