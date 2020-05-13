import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy.stats import skew

from BERT.run_classifier_multipleParameter import InputExample
from util.load_datasets import load_conll, AZ_loader, AQ_loader, AR_loader, ISA_loader, load_comp


def ArgRecognition_Analysis():
    def _get_examples_GM(descr):
        if descr == "Train":
            c = AR_loader.get_ArgRecognition_GM_dataset(1)
        elif descr == "Dev":
            c = AR_loader.get_ArgRecognition_GM_dataset(2)
        else:
            c = AR_loader.get_ArgRecognition_GM_dataset(3)
        return c

    def _get_examples_UGIP(descr):
        if descr == "Train":
            c = AR_loader.get_ArgRecognition_UGIP_dataset(1)
        elif descr == "Dev":
            c = AR_loader.get_ArgRecognition_UGIP_dataset(2)
        else:
            c = AR_loader.get_ArgRecognition_UGIP_dataset(3)
        return c

    def analyse_GM():
        print("Statistics about Dataset Gay Marriage")

        train = _get_examples_GM("Train")
        dev = _get_examples_GM("Dev")
        test = _get_examples_GM("Test")
        all_InputExamples = pd.concat((train, test, dev), axis=0)
        print("all Examples ", len(all_InputExamples))
        print("Statistics:")

        length_comment_list = []
        exceeding_text_list = []
        idx = 0
        for entry in all_InputExamples.comment_text:

            text_length = len(entry.split(" "))
            if text_length > 128:
                length_comment_list.append(text_length - 128)
                exceeding_text_list.append(entry)
            idx += 1
        # ' count all unique comments
        #     length_comment_list.append(text_length)
        #     exceeding_text_list.append(entry)
        #
        # unique_comment_text = np.unique(exceeding_text_list)
        # print("length unique all texts", len(unique_comment_text))

        list.sort(length_comment_list, reverse=True)
        print("Top 10 Lenths:", length_comment_list[:10])

        print("Max Comment", max(length_comment_list))
        print("Avg Comment", np.mean(length_comment_list))
        print("min Comment", min(length_comment_list))

        plt.style.use('ggplot')
        plt.title("Comment Length UGIP - Exceeding ")
        plt.hist(length_comment_list, bins='auto')
        plt.show()
        print(skew(length_comment_list))

        sns.set_style('darkgrid')
        sns.distplot(length_comment_list).set_title("Comment Length - Exceeding")
        plt.show()

        # for ele in exceeding_text_list:
        #     print(ele[128:] + "\n")

        unique_comment_text = np.unique(exceeding_text_list)
        print("length unique exceedings texts", len(unique_comment_text))

        # for unique_ele in unique_comment_text:
        #     print(" ".join(unique_ele.split(" ")[128:]))

    def analyse_UGIP():
        print("Statistics about Dataset Under God in Plege")
        train = _get_examples_UGIP("Train")
        dev = _get_examples_UGIP("Dev")
        test = _get_examples_UGIP("Test")
        all_InputExamples = pd.concat((train, test, dev), axis=0)
        print("all Examples ", len(all_InputExamples))
        print("Statistics:")

        length_comment_list = []
        exceeding_text_list = []
        idx = 0
        for entry in all_InputExamples.comment_text:

            text_length = len(entry.split(" "))
            if text_length > 128:
                length_comment_list.append(text_length - 128)
                exceeding_text_list.append(entry)
            idx += 1
        #     length_comment_list.append(text_length)
        #     exceeding_text_list.append(entry)
        #
        # unique_comment_text = np.unique(exceeding_text_list)
        # print("length unique all texts", len(unique_comment_text))

        list.sort(length_comment_list, reverse=True)
        print("Top 10 Lenths:", length_comment_list[:10])

        print("Max Comment", max(length_comment_list))
        print("Avg Comment", np.mean(length_comment_list))
        print("min Comment", min(length_comment_list))

        plt.style.use('ggplot')
        plt.title("Comment Length UGIP - Exceeding ")
        plt.hist(length_comment_list, bins='auto')
        plt.show()
        print(skew(length_comment_list))

        sns.set_style('darkgrid')
        sns.distplot(length_comment_list).set_title("Comment Length - Exceeding")
        plt.show()

        # for ele in exceeding_text_list:
        #     print (ele[128:] + "\n")

        unique_comment_text = np.unique(exceeding_text_list)
        print("length unique exceedings texts", len(unique_comment_text))

        # for unique_ele in unique_comment_text:
        #     print(" ".join(unique_ele.split(" ")[128:]))

    analyse_GM()
    # analyse_UGIP()


def InsufficientArgSupport_Analysis():
    def _get_examples(descr):
        if descr == "Train":
            c = ISA_loader.get_InsuffientSupport_datset_byFilter(case=1)
        elif descr == "Dev":
            c = ISA_loader.get_InsuffientSupport_datset_byFilter(case=2)
        else:
            c = ISA_loader.get_InsuffientSupport_datset_byFilter(case=3)
        return c

    train = _get_examples("Train")
    dev = _get_examples("Dev")
    test = _get_examples("Test")
    all_InputExamples = pd.concat((train, test, dev), axis=0)
    print("Statistics  ISA >128:")

    length_list = []
    for sentence in all_InputExamples.TEXT:
        length_list.append(len(sentence.split(" ")))
    length_list = [length - 128 for length in length_list if length > 128]

    print("# Sentences", len(length_list))
    print("Max", max(length_list))
    print("Avg", np.mean(length_list))
    print("Min", min(length_list))
    list.sort(length_list, reverse=True)
    print("Top 10 Lenths:", length_list[:10])
    print("skew", skew(length_list))
    plt.style.use('ggplot')
    plt.title(' ISA Exceeding Text')
    plt.hist(length_list, bins='auto')
    plt.show()

    sns.set_style('darkgrid')
    sns.distplot(length_list).set_title("ISA TEXT")
    plt.show()


def main():
    # ArgRecognition_Analysis()
    InsufficientArgSupport_Analysis()


if __name__ == "__main__":
    main()
