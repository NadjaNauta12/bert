import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy.stats import skew

from BERT.run_classifier_multipleParameter import InputExample
from util.load_datasets import load_conll, AZ_loader, AQ_loader, AR_loader, ISA_loader, load_comp
from collections import Counter
from util.PathMux import get_path_to_BERT, get_path_to_OS
from BERT import tokenization


def ArgQuality_Analysis():
    limit = 125

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
    all_InputExamples['completeSequence'] = all_InputExamples['a1'] + all_InputExamples['a2']
    print("Statistics:")
    print(all_InputExamples.label.value_counts())

    # Statistics for each argument  - better to do it alltogether
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
    plt.axvline(limit, 0, color='blue')
    plt.hist(length_a1_list, bins='auto')
    plt.title("AQ - Argument 1")
    plt.show()
    print(skew(length_a1_list))

    plt.style.use('ggplot')
    plt.hist(length_a2_list, bins='auto')
    plt.title("AQ - Argument 2")
    plt.show()
    print(skew(length_a2_list))

    sns.set_style('darkgrid')
    sns.distplot(length_a1_list).set_title("AQ Arg 1 Length")
    plt.show()
    sns.set_style('darkgrid')
    sns.distplot(length_a2_list).set_title("AQ Arg2 Length")
    plt.show()

    completeSequence_length_list = []
    for entry in all_InputExamples.completeSequence:
        completeSequence_length_list.append(len(entry.split(" ")))
    print("Max ", max(completeSequence_length_list))
    print("Avg ", np.mean(completeSequence_length_list))

    plt.style.use('ggplot')
    plt.axvline(limit, 0, color='blue')
    plt.hist(completeSequence_length_list, bins='auto')
    plt.title("AQ - Complete Sequence Lengths")
    plt.show()
    print(skew(completeSequence_length_list))

    sns.set_style('darkgrid')
    sns.distplot(completeSequence_length_list).set_title("Sequence Lengths AQ")
    plt.show()

    _word_piece_Sequencelength_Analysis(all_InputExamples, "AQ WP Tokenization", "a1", "a2", limit)


def ArgZoningI_Analysis():
    limit = 126

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
    print("Distribution", all_InputExamples.AZ_category.value_counts())

    length_list = []
    for sentence in all_InputExamples.text:
        length_list.append(len(sentence.split(" ")))
    print("Max", max(length_list))
    print("Avg", np.mean(length_list))
    list.sort(length_list, reverse=True)
    print(length_list)

    plt.style.use('ggplot')
    plt.axvline(limit, 0, color='blue')
    plt.hist(length_list, bins='auto')
    plt.title("Argument Zoning - Sentence Length")
    plt.show()
    print(skew(length_list))

    sns.set_style('darkgrid')
    sns.distplot(length_list).set_title("AZ Sentence Length")
    plt.show()

    _word_piece_Sequencelength_Analysis(all_InputExamples, "AZ WP Tokenization", "text", None, limit)


def ArgRecognition_Analysis():
    limit = 125

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

        list.sort(length_comment_list, reverse=True)
        print("Top 10 Lenths:", length_comment_list[:10])

        exceeding_counter = [ele for ele in length_comment_list if ele > limit]
        print("Examples exceeding ", limit, len(exceeding_counter))

        plt.style.use('ggplot')
        plt.title("Comment Length  GM")
        plt.hist(length_comment_list, bins='auto')
        plt.axvline(limit, 0, color='blue')
        plt.show()
        print(skew(length_comment_list))

        plt.style.use('ggplot')
        plt.title("Argument Length- GM")
        plt.axvline(limit, 0, color='blue')
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
            if "Wow" in entry:
                print(len(entry.split(" ")))
        for entry in all_InputExamples.argument_text:
            length_argument_list.append(len(entry.split(" ")))
        print("Max Comment", max(length_comment_list))
        print("Avg Comment", np.mean(length_comment_list))
        print("Max Argument", max(length_argument_list))
        print("Avg Argument", np.mean(length_argument_list))

        list.sort(length_comment_list, reverse=True)
        print("Top 10 Lenths:", length_comment_list[:10])

        exceeding_counter = [ele for ele in length_comment_list if ele > limit]
        print("Examples exceeding ", limit, len(exceeding_counter))

        plt.style.use('ggplot')
        plt.title("Comment Length  UGIP")
        plt.hist(length_comment_list, bins='auto')
        plt.axvline(limit, 0, color='blue')
        plt.show()
        print(skew(length_comment_list))

        plt.style.use('ggplot')
        plt.title("Argument Length- UGIP")
        plt.hist(length_argument_list, bins='auto')
        plt.axvline(limit, 0, color='blue')
        plt.show()
        print(skew(length_argument_list))

        sns.set_style('darkgrid')
        sns.distplot(length_comment_list).set_title("Comment Length UGIP")
        plt.show()
        sns.set_style('darkgrid')
        sns.distplot(length_argument_list).set_title("Argument Length UGIP")
        plt.show()


    def analyse_GM_complete_Sequence():
        print("Statistics about Dataset Gay Marriage - Complete Input Sequence")
        train = _get_examples_GM("Train")
        dev = _get_examples_GM("Dev")
        test = _get_examples_GM("Test")
        all_InputExamples = pd.concat((train, test, dev), axis=0)
        print("all Examples ", len(all_InputExamples))
        all_InputExamples['completeSequence'] = all_InputExamples['comment_text'] +  [" "]*len(all_InputExamples) +all_InputExamples['argument_text']
        print("Statistics:")
        print(all_InputExamples.label.value_counts())

        length_completeSequence_list = []
        for entry in all_InputExamples.completeSequence:
            length_completeSequence_list.append(len(entry.split(" ")))

        print("Max ", max(length_completeSequence_list))
        print("Avg ", np.mean(length_completeSequence_list))

        list.sort(length_completeSequence_list, reverse=True)
        print("Top 10 Lenths:", length_completeSequence_list[:10])

        exceeding_counter = [ele for ele in length_completeSequence_list if ele > limit]
        print("Examples exceeding ", limit, len(exceeding_counter))

        plt.style.use('ggplot')
        plt.title("Sequence Length  GM")
        plt.hist(length_completeSequence_list, bins='auto')
        plt.axvline(limit, 0, color='blue')
        #plt.show()
        print(skew(length_completeSequence_list))
        plt.clf()
        sns.set_style('whitegrid', {'grid.linestyle': '--'})
        chart = sns.distplot(length_completeSequence_list, kde=False, hist=True, color="darkslategrey")
        chart.set_title("Complete Sequence Lengths - Gay Marriage (AR)")
        chart.set(xlabel='Sequence Length', ylabel='Count of Occurrences')
        chart.spines['right'].set_visible(False)
        chart.spines['top'].set_visible(False)
        plt.axvline(limit, 0, color='black',linestyle='-.')#, '-.', ':'])
        plt.show()

        _word_piece_Sequencelength_Analysis(all_InputExamples, "Input Lengths with Word Pieces - AR Gay Marriage", "comment_text",
                                            "argument_text", limit)

    def analyse_UGIP_complete_Sequence():
        print("Statistics about Dataset Gay Marriage - Complete Input Sequence")
        train = _get_examples_UGIP("Train")
        dev = _get_examples_UGIP("Dev")
        test = _get_examples_UGIP("Test")
        all_InputExamples = pd.concat((train, test, dev), axis=0)
        print("all Examples ", len(all_InputExamples))
        all_InputExamples['completeSequence'] = all_InputExamples['comment_text'] + [" "]*len(all_InputExamples) +  all_InputExamples[
            'argument_text']
        print("Statistics:")
        print(all_InputExamples.label.value_counts())

        length_completeSequence_list = []
        for entry in all_InputExamples.completeSequence:
            length_completeSequence_list.append(len(entry.split(" ")))
            if len(entry.split(" ")) == 250:
                print("hey")

        print("Max ", max(length_completeSequence_list))
        print("Avg ", np.mean(length_completeSequence_list))

        list.sort(length_completeSequence_list, reverse=True)
        print("Top 10 Lenths:", length_completeSequence_list[:10])

        exceeding_counter = [ele for ele in length_completeSequence_list if ele > limit]
        print("Examples exceeding ", limit, len(exceeding_counter))

        plt.style.use('ggplot')
        plt.title("Sequence Length  UGIP")
        plt.hist(length_completeSequence_list, bins='auto')
        plt.axvline(limit, 0, color='blue')
        #plt.show()
        print(skew(length_completeSequence_list))

        sns.set_style('darkgrid')
        sns.distplot(length_completeSequence_list).set_title("Sequence Length")
        #plt.show()

        plt.clf()
        sns.set_style('whitegrid', {'grid.linestyle': '--'})
        chart = sns.distplot(length_completeSequence_list, kde=False, hist=True, color="darkslategrey")
        chart.set_title("Complete Sequence Lengths - Under God in Pledge (AR)")
        chart.set(xlabel='Sequence Length', ylabel='Count of Occurrences')
        chart.spines['right'].set_visible(False)
        chart.spines['top'].set_visible(False)
        plt.axvline(limit, 0, color='black',linestyle='-.')#, '-.', ':'])
        plt.show()

        _word_piece_Sequencelength_Analysis(all_InputExamples, "Input Lengths with Word Pieces - AR Under God in Pledge", "comment_text",
                                            "argument_text", limit)

    #analyse_GM()
    analyse_GM_complete_Sequence()
    # analyse_UGIP()
    analyse_UGIP_complete_Sequence()


def InsufficientArgSupport_Analysis():
    limit = 126

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
    # print("Distribution", all_InputExamples.ANNOTATION.value_counts())

    length_list = []
    for sentence in all_InputExamples.TEXT:
        length_list.append(len(sentence.split(" ")))
    print("# Sentences", len(length_list))
    print("Max", max(length_list))
    print("Avg", np.mean(length_list))
    list.sort(length_list, reverse=True)
    print("Top 10 Lenths:", length_list[:10])

    above_limit = []
    for ele in length_list:
        if ele > limit:
            above_limit.append(ele)
    print(len(above_limit))

    plt.style.use('ggplot')
    plt.title(' ISA Text')
    plt.axvline(limit, 0, color='blue')
    plt.hist(length_list, bins='auto')
    plt.show()
    print(skew(length_list))

    sns.set_style('darkgrid')
    sns.distplot(length_list).set_title("ISA TEXT")
    plt.show()

    plt.clf()
    sns.set_style('whitegrid', {'grid.linestyle': '--'})
    chart = sns.distplot(length_list, kde=False, hist=True, color="darkslategrey")
    chart.set_title("Complete Sequence Lengths  - ISA")
    chart.set(xlabel='Sequence Length', ylabel='Count of Occurrences')
    chart.spines['right'].set_visible(False)
    chart.spines['top'].set_visible(False)
    plt.axvline(limit, 0, color='black', linestyle='-.')  # , '-.', ':'])
    plt.show()

    _word_piece_Sequencelength_Analysis(all_InputExamples, "Input Lengths with Word Pieces - ISA", "TEXT", None, limit)


def ACI_Stab_Analysis():
    limit = 126

    def _get_examples(descr):
        data_dir = get_path_to_OS() + "/data/Argument_Component_Identification_Stab"
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

    print("Statistics about Dataset ACI-Stab")
    train = _get_examples("Train")
    dev = _get_examples("Dev")
    test = _get_examples("Test")
    label_list = []

    sentences = []
    sentence_lengths = []
    for ele in train:
        words = [list[0] for list in ele]
        sentences.append(" ".join(words))
        sentence_lengths.append(len(words))
        labels = [list[1] for list in ele]
        label_list += labels
    for ele in dev:
        words = [list[0] for list in ele]
        sentences.append(" ".join(words))
        sentence_lengths.append(len(words))
        labels = [list[1] for list in ele]
        label_list += labels
    for ele in test:
        words = [list[0] for list in ele]
        sentences.append(" ".join(words))
        sentence_lengths.append(len(words))
        labels = [list[1] for list in ele]
        label_list += labels

    print("all Examples ", len(sentence_lengths))
    print("Statistics:")

    count_values = Counter(label_list)
    print("Distribution labels", count_values)

    print("Max ", max(sentence_lengths))
    print("Avg ", np.mean(sentence_lengths))

    plt.style.use('ggplot')
    plt.title("Sentence Length ACI Stab")
    plt.hist(sentence_lengths, bins='auto')
    plt.axvline(limit, 0, color='blue')
    plt.show()
    print(skew(sentence_lengths))

    sns.set_style('darkgrid')
    sns.distplot(sentence_lengths).set_title("Sentence Length ACI Stab")
    plt.show()

    _word_piece_Sequencelength_Analysis(pd.DataFrame(sentences, columns=["TEXT"]),
                                        "ACI_Stab WP Tokenization", "TEXT", None, limit)


def ACI_Lauscher_Analysis():
    limit = 126

    def _get_examples(descr):
        data_dir = get_path_to_OS() + "/data/Argument_Component_Identification_Lauscher/annotations_conll_all_splitted"
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

    print("Statistics about Dataset ACI-Lauscher")
    train = _get_examples("Train")
    dev = _get_examples("Dev")
    test = _get_examples("Test")

    sentences = []
    sentence_lengths = []
    for ele in train:
        words = [list[0] for list in ele]
        sentences.append(" ".join(words))
        sentence_lengths.append(len(words))
    for ele in dev:
        words = [list[0] for list in ele]
        sentences.append(" ".join(words))
        sentence_lengths.append(len(words))
    for ele in test:
        words = [list[0] for list in ele]
        sentences.append(" ".join(words))
        sentence_lengths.append(len(words))




    print("all Examples ", len(sentence_lengths))
    print("Statistics:")

    print("Max ", max(sentence_lengths))
    print("Avg ", np.mean(sentence_lengths))
    list.sort(sentence_lengths, reverse=True)
    #sentence_lengths = sentence_lengths[3:]
    print(sentence_lengths[:20])
    print("shortest length")
    list.sort(sentence_lengths, reverse=False)
    print(sentence_lengths[:1])

    # plt.style.use('ggplot')
    # plt.title("Sentence Length ACI Lauscher")
    # plt.hist(sentence_lengths, bins='auto')
    # plt.axvline(limit, 0, color='blue')
    # plt.show()
    # print(skew(sentence_lengths))

    #sns.set_style('darkgrid')
    #sns.distplot(sentence_lengths).set_title("Sentence Length ACI Lauscher")
    #plt.show()

    # plt.clf()
    # sns.set_style('whitegrid', {'grid.linestyle': '--'})
    # chart = sns.distplot(sentence_lengths, kde=False, hist=True, color="darkslategrey")
    # chart.set_title("Complete Sequence Lengths- ACI Corpus Laucher et al.")
    # chart.set(xlabel='Sequence Length', ylabel='Count of Occurrences')
    # chart.spines['right'].set_visible(False)
    # chart.spines['top'].set_visible(False)
    # plt.axvline(limit, 0, color='black', linestyle='-.')  # , '-.', ':'])
    # plt.show()


    above_max_seq = [length for length in sentence_lengths if length > limit]
    # print("# Instance above", limit, len(above_max_seq))
    # plt.style.use('ggplot')
    # plt.title("Sentence Length ACI Lauscher - Above Seq. Length only")
    # plt.hist(above_max_seq, bins='auto')
    # plt.axvline(limit, 0, color='blue')
    # plt.show()
    # print(skew(above_max_seq))
    #
    #
    # plt.clf()
    # sns.set_style('whitegrid', {'grid.linestyle': '--'})
    # chart = sns.distplot(above_max_seq, kde=False, hist=True, color="darkslategrey")
    # chart.set_title("Sequence Lengths Above Limit - ACI Corpus Laucher et al.")
    # chart.set(xlabel='Sequence Length', ylabel='Count of Occurrences')
    # chart.spines['right'].set_visible(False)
    # chart.spines['top'].set_visible(False)
    # plt.axvline(limit, 0, color='black', linestyle='-.')  # , '-.', ':'])
    # plt.show()


    _word_piece_Sequencelength_Analysis(pd.DataFrame(sentences, columns=["TEXT"]),
                                         "Input Lengths with Word Pieces - ACI  Lauscher et al.", "TEXT", None, limit)


def _word_piece_Sequencelength_Analysis(data, text, col1, col2, limit):
    if isinstance(data, pd.DataFrame) and col1 is not None and col2 is not None:
        input_a = data[col1]
        input_b = data[col2]
        assert len(input_a) == len(input_b)

        BERT_path = get_path_to_BERT() + "/vocab.txt"
        tokenizer = tokenization.FullTokenizer(vocab_file=BERT_path, do_lower_case=True)

        instances_above_limit = 0
        wp_lengths = []
        for i, ele in enumerate(input_a):
            length = len(tokenizer.tokenize(input_a.iloc[i])) + len(tokenizer.tokenize(input_b.iloc[i]))
            wp_lengths.append(length)
            if length > limit:
                instances_above_limit += 1

        print("Max ", max(wp_lengths))
        print("Avg ", np.mean(wp_lengths))
        print("Instances above limit:", instances_above_limit)

        plt.style.use('ggplot')
        plt.axvline(limit, 0, color='blue')
        plt.hist(wp_lengths, bins='auto')
        plt.title(text)
        plt.show()
        print("Skew ", text, skew(wp_lengths))

        # nicer image
        plt.clf()
        sns.set_style('whitegrid', {'grid.linestyle': '--'})
        chart = sns.distplot(wp_lengths, kde=False, hist=True, color="darkslategrey")
        chart.set_title(text)
        chart.set(xlabel='Complete Input Lengths', ylabel='Count of Occurrences')
        chart.spines['right'].set_visible(False)
        chart.spines['top'].set_visible(False)
        plt.axvline(limit, 0, color='black', linestyle='-.')  # , '-.', ':'])
        plt.show()

    elif isinstance(data, pd.DataFrame) and col1 is not None and col2 is None:
        input_a = data[col1]

        BERT_path = get_path_to_BERT() + "/vocab.txt"
        tokenizer = tokenization.FullTokenizer(vocab_file=BERT_path, do_lower_case=True)
        instances_above_limit = 0
        wp_lengths = []
        for i, ele in enumerate(input_a):
            length = len(tokenizer.tokenize(input_a.iloc[i]))
            wp_lengths.append(length)
            if length > limit:
                instances_above_limit += 1

        list.sort(wp_lengths, reverse=True)
        print("Max ", max(wp_lengths))
        print("Avg ", np.mean(wp_lengths))
        print("Instances above limit:", instances_above_limit)

        plt.style.use('ggplot')
        plt.axvline(limit, 0, color='blue')
        plt.hist(wp_lengths, bins='auto')
        plt.title(text)
        plt.show()
        print("Skew ", text, skew(wp_lengths))


        # for Lauscher Analysis
        if "Lauscher" in text:
            wp_lengths = [length for length in wp_lengths if length < limit]

        # nicer image
        plt.clf()
        sns.set_style('whitegrid', {'grid.linestyle': '--'})
        chart = sns.distplot(wp_lengths, kde=False, hist=True, color="darkslategrey")
        chart.set_title(text)
        chart.set(xlabel='Complete Input Lengths', ylabel='Count of Occurrences')
        chart.spines['right'].set_visible(False)
        chart.spines['top'].set_visible(False)
        plt.axvline(limit, 0, color='black', linestyle='-.')  # , '-.', ':'])
        plt.show()


def main():
    # Double Check marked with ###
    # ArgQuality_Analysis()  # not affected
    # ArgZoningI_Analysis()
    #ArgRecognition_Analysis()
    #InsufficientArgSupport_Analysis()

    ### ACI_Stab_Analysis() #  not affected
    ACI_Lauscher_Analysis()


if __name__ == "__main__":
    main()
