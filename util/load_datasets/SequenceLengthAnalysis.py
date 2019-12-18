import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy.stats import skew

from BERT.run_classifier_multipleParameter import InputExample
from util.load_datasets import load_conll, AZ_loader, AQ_loader, AR_loader, ISA_loader, load_comp


def ArgQuality_Analysis():
    def _get_examples(descr):
        if descr == "Train":
            c = AQ_loader.load_ArgQuality_datset(case_ID=1)
        elif descr == "Dev":
            c = AQ_loader.load_ArgQuality_datset(case_ID=2)
        else:
            c = AQ_loader.load_ArgQuality_datset(case_ID=3)
        return c

    train = _get_examples("Train")
    dev = _get_examples("Dev")
    test = _get_examples("Test")
    all_InputExamples = pd.concat((train, test, dev), axis=0)
    print("all Examples ", len(all_InputExamples))
    print("Statistics:")
    print(all_InputExamples.label.value_counts())

    length_a1_list = []
    length_a2_list = []
    for entry in all_InputExamples.a1:
        length_a1_list.append(len(entry.split(" ")))
    for entry in all_InputExamples.a2:
        length_a2_list.append(len(entry.split(" ")))
    print("Max A1", max(length_a1_list))
    print("Avg A1", np.mean(length_a1_list))
    print("Max A2", max(length_a2_list))
    print("Avg A2", np.mean(length_a2_list))

    plt.style.use('ggplot')
    plt.axvline(128, 0, color='blue')
    plt.hist(length_a1_list, bins='auto')
    plt.show()
    print(skew(length_a1_list))

    plt.style.use('ggplot')
    plt.hist(length_a2_list, bins='auto')
    plt.show()
    print(skew(length_a2_list))

    sns.set_style('darkgrid')
    sns.distplot(length_a1_list)
    plt.show()
    sns.set_style('darkgrid')
    sns.distplot(length_a2_list)
    plt.show()


def ArgZoningI_Analysis():
    def _get_examples(descr):
        if descr == "Train":
            c = AZ_loader.get_ArgZoning_dataset(case=1)
        elif descr == "Dev":
            c = AZ_loader.get_ArgZoning_dataset(case=2)
        else:
            c = AZ_loader.get_ArgZoning_dataset(case=3)
        return c

    train = _get_examples("Train")
    dev = _get_examples("Dev")
    test = _get_examples("Test")
    all_InputExamples = pd.concat((train, test, dev), axis=0)
    print("all Examples ", len(all_InputExamples))
    print("Statistics:")
    print(all_InputExamples.text.value_counts())

    length_list = []
    for sentence in all_InputExamples.text:
        length_list.append(len(sentence.split(" ")))
    print("Max", max(length_list))
    print("Avg", np.mean(length_list))
    print(list.sort(length_list, reverse=True))

    plt.style.use('ggplot')
    plt.axvline(128, 0, color='blue')
    plt.hist(length_list, bins='auto')
    plt.show()
    print(skew(length_list))

    sns.set_style('darkgrid')
    sns.distplot(length_list)
    plt.show()


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
        print(all_InputExamples.label.value_counts())

        length_comment_list = []
        length_argument_list = []
        for entry in all_InputExamples.comment_text:
            length_comment_list.append(len(entry.split(" ")))
        for entry in all_InputExamples.argument_text:
            length_argument_list.append(len(entry.split(" ")))
        print("Max Comment", max(length_comment_list))
        print("Avg Comment", np.mean(length_comment_list))
        print("Max Argument", max(length_argument_list))
        print("Avg Argument", np.mean(length_argument_list))

        plt.style.use('ggplot')
        plt.title("Comment Length  GM")
        plt.hist(length_comment_list, bins='auto')
        plt.axvline(128, 0, color='blue')
        plt.show()
        print(skew(length_comment_list))

        plt.style.use('ggplot')
        plt.title("Argument Length- GM")
        plt.hist(length_argument_list, bins='auto')
        plt.show()
        print(skew(length_argument_list))

        sns.set_style('darkgrid')
        sns.distplot(length_comment_list).set_title("Comment Length")
        plt.show()
        sns.set_style('darkgrid')
        sns.distplot(length_argument_list).set_title("Argument Length")
        plt.show()

    def analyse_UGIP():
        print("Statistics about Dataset Under God in Plege")
        train = _get_examples_UGIP("Train")
        dev = _get_examples_UGIP("Dev")
        test = _get_examples_UGIP("Test")
        all_InputExamples = pd.concat((train, test, dev), axis=0)
        print("all Examples ", len(all_InputExamples))
        print("Statistics:")
        print(all_InputExamples.label.value_counts())

        length_comment_list = []
        length_argument_list = []
        for entry in all_InputExamples.comment_text:
            length_comment_list.append(len(entry.split(" ")))
        for entry in all_InputExamples.argument_text:
            length_argument_list.append(len(entry.split(" ")))
        print("Max Comment", max(length_comment_list))
        print("Avg Comment", np.mean(length_comment_list))
        print("Max Argument", max(length_argument_list))
        print("Avg Argument", np.mean(length_argument_list))

        plt.style.use('ggplot')
        plt.title("Comment Length  UGIP")
        plt.hist(length_comment_list, bins='auto')
        plt.axvline(128, 0, color='blue')
        plt.show()
        print(skew(length_comment_list))

        plt.style.use('ggplot')
        plt.title("Argument Length- UGIP")
        plt.hist(length_argument_list, bins='auto')
        plt.show()
        print(skew(length_argument_list))

        sns.set_style('darkgrid')
        sns.distplot(length_comment_list).set_title("Comment Length UGIP")
        plt.show()
        sns.set_style('darkgrid')
        sns.distplot(length_argument_list).set_title("Argument Length UGIP")
        plt.show()

    analyse_GM()
    analyse_UGIP()


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
    print("all Examples ", len(all_InputExamples))
    print("Statistics:")
    print(all_InputExamples.TEXT.value_counts())

    length_list = []
    for sentence in all_InputExamples.TEXT:
        length_list.append(len(sentence.split(" ")))
    print("Max", max(length_list))
    print("Avg", np.mean(length_list))
    list.sort(length_list, reverse=True)
    print("Top 10 Lenths:", length_list[:10])

    plt.style.use('ggplot')
    plt.title(' ISA Text')
    plt.axvline(128, 0, color='blue')
    plt.hist(length_list, bins='auto')
    plt.show()
    print(skew(length_list))

    sns.set_style('darkgrid')
    sns.distplot(length_list).set_title("ISA TEXT")
    plt.show()


def ACI_Habernal_Analysis():
    def _get_examples(descr):
        data_dir = "C:/Users/Wifo/PycharmProjects/Masterthesis/data/Argument_Component_Identification_Habernal"
        if descr == "Train":
            path = data_dir + '/train'
            c = load_comp.parse_comp_files(path=path)
        elif descr == "Dev":
            path = data_dir + '/dev'
            c = load_comp.parse_comp_files(path=path)
        elif descr == "Test":
            path = data_dir + '/test'
            c = load_comp.parse_comp_files(path=path)
        else:
            c = None

        flat_sentences = [item for sublist in c for item in sublist]
        return flat_sentences

    def _flat_sentences_to_InputExamples(flat_sentences, descr):
        for ele in flat_sentences:
            words = [list[0] for list in ele]
            test = ' '.join(words)
            if (len(words) > 128):
                counter_Seq += 1

            labels = [list[1] for list in ele]
            most_frequent = max(set(labels), key=labels.count)
            # print ("MOST FREQ. Label", most_frequent)
            guid = descr + "-" + str(counter)
            examples_list.append(InputExample(guid=guid, text_a=' '.join(words), text_b=None, label=most_frequent))
            counter += 1
        return examples_list

    print("Statistics about Dataset ACI-Habernal")
    train = _get_examples("Train")
    dev = _get_examples("Dev")
    test = _get_examples("Test")

    sentence_lengths = []
    for ele in train:
        words = [list[0] for list in ele]
        sentence_lengths.append(len(words))
    for ele in dev:
        words = [list[0] for list in ele]
        sentence_lengths.append(len(words))
    for ele in test:
        words = [list[0] for list in ele]
        sentence_lengths.append(len(words))

    print("all Examples ", len(sentence_lengths))
    print("Statistics:")

    print("Max ", max(sentence_lengths))
    print("Avg ", np.mean(sentence_lengths))


    plt.style.use('ggplot')
    plt.title("Sentence Length ACI Habernal")
    plt.hist(sentence_lengths, bins='auto')
    plt.axvline(128, 0, color='blue')
    plt.show()
    print(skew(sentence_lengths))

    sns.set_style('darkgrid')
    sns.distplot(sentence_lengths).set_title("Sentence Length ACI Habernal")
    plt.show()


def ACI_Lauscher_Analysis():
    def _get_examples(descr):
        data_dir = "C:/Users/Wifo/PycharmProjects/Masterthesis/data/Argument_Component_Identification_Lauscher/annotations_conll_all_splitted"
        if descr == "Train":
            path = data_dir + '/train'
            c = load_conll.parse_conll_files(path=path)
        elif descr == "Dev":
            path = data_dir + '/dev'
            c = load_conll.parse_conll_files(path=path)
        elif descr == "Test":
            path = data_dir + '/test'
            c = load_conll.parse_conll_files(path=path)
        else:
            c = None

        flat_sentences = [item for sublist in c for item in sublist]
        return flat_sentences

    print("Statistics about Dataset ACI-Habernal")
    train = _get_examples("Train")
    dev = _get_examples("Dev")
    test = _get_examples("Test")

    sentence_lengths = []
    for ele in train:
        words = [list[0] for list in ele]
        sentence_lengths.append(len(words))
    for ele in dev:
        words = [list[0] for list in ele]
        sentence_lengths.append(len(words))
    for ele in test:
        words = [list[0] for list in ele]
        sentence_lengths.append(len(words))

    print("all Examples ", len(sentence_lengths))
    print("Statistics:")

    print("Max ", max(sentence_lengths))
    print("Avg ", np.mean(sentence_lengths))
    list.sort(sentence_lengths, reverse=True)
    print(sentence_lengths[:20])

    plt.style.use('ggplot')
    plt.title("Sentence Length ACI Lauscher")
    plt.hist(sentence_lengths, bins='auto')
    plt.axvline(128, 0, color='blue')
    plt.show()
    print(skew(sentence_lengths))

    sns.set_style('darkgrid')
    sns.distplot(sentence_lengths).set_title("Sentence Length ACI Lauscher")
    plt.show()


    above_max_seq = [ length for length in sentence_lengths if length >128]
    plt.style.use('ggplot')
    plt.title("Sentence Length ACI Lauscher - Above Seq. Length only")
    plt.hist(above_max_seq, bins='auto')
    plt.axvline(128, 0, color='blue')
    plt.show()
    print(skew(above_max_seq))

    sns.set_style('darkgrid')
    sns.distplot(above_max_seq).set_title("Sentence Length ACI Lauscher - Above Seq. Length only")
    plt.show()

def main():
    # ArgQuality_Analysis()
    # ArgZoningI_Analysis()
    # ArgRecognition_Analysis()
    # InsufficientArgSupport_Analysis()
    # not affected
    ACI_Habernal_Analysis()
    # ACI_Lauscher_Analysis()


if __name__ == "__main__":
    main()
