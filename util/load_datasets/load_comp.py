"""
Referring to  A. Lauscher
https://raw.githubusercontent.com/anlausch/multitask_sciarg/master/load_conll.py
"""
import os
import codecs
import numpy as np


def parse_comp_file(file, multiple=False):
    tokens = []
    sentences = []
    for line in file.readlines():
        if (line == "\n") or (line == "\r\n"):
            if len(tokens) != 0:
                # sentence = CoNLL_Sentence(tokens=tokens)
                sentences.append(tokens)  # Req. separator between sentences'''
                tokens = []
            else:
                print("That should not happen.")
        else:
            parts = line.split("\t")
            if multiple == False:
                token = [parts[0], parts[1].strip()]  # TODO currently only contains two parts token + tokenlabel, parts[ 2]]
                # Simplified_CoNLL_Token(token=parts[0], token_label=parts[1], sentence_label=parts[2])
            else:
                token = [parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]]
                print("Should not load multiple columns - only two")
            tokens.append(token)
    return sentences


def parse_comp_files(path, multiple=False):
    sentences = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            with codecs.open(os.path.join(subdir, file), "r", "utf8") as f:
                file_sentences = parse_comp_file(f, multiple=multiple)
                sentences.append(file_sentences)
    return sentences


def transform_to_model_input(sentences):
    x = []
    y_arg = []
    #y_rhet = []
    for sentence in sentences:
        x_sentence = []
        y_sentence_arg = []
        #y_sentence_rhet = []
        for token in sentence:
            x_sentence.append(token[0])
            y_sentence_arg.append(token[1])
            #y_sentence_rhet.append(token[2])
        x.append(np.array(x_sentence))
        y_arg.append(np.array(y_sentence_arg))
        #y_rhet.append(np.array(y_sentence_rhet))
    return np.array(x), np.array(y_arg)#, np.array(y_rhet)


def transform_to_model_input_multiple(sentences):
    x = []
    y_arg = []
    y_rhet = []
    y_aspect = []
    y_summary = []
    y_citation = []
    for sentence in sentences:
        x_sentence = []
        y_sentence_arg = []
        y_sentence_rhet = []
        y_sentence_aspect = []
        y_sentence_summary = []
        y_sentence_citation = []
        for token in sentence:
            x_sentence.append(token[0])
            y_sentence_arg.append(token[1])
            y_sentence_rhet.append(token[2])
            y_sentence_aspect.append(token[3])
            y_sentence_summary.append(token[4])
            y_sentence_citation.append(token[5])
        x.append(np.array(x_sentence))
        y_arg.append(np.array(y_sentence_arg))
        y_rhet.append(np.array(y_sentence_rhet))
        y_aspect.append(np.array(y_sentence_aspect))
        y_summary.append(np.array(y_sentence_summary))
        y_citation.append(np.array(y_sentence_citation))
    return np.array(x), np.array(y_arg), np.array(y_rhet), np.array(y_aspect), np.array(y_summary), np.array(y_citation)


def load_data(path):
    sentences = parse_comp_files(path)
    flat_sentences = [item for sublist in sentences for item in sublist]
    x, y_arg, y_rhet = transform_to_model_input(flat_sentences)
    print("Data size: " + str(len(x)))
    return x, y_arg, y_rhet


def load_data_multiple(path=""):
    sentences = parse_comp_files(path, multiple=False)  # TODO changes Multiple to false
    flat_sentences = [item for sublist in sentences for item in sublist]
    x, y_arg, y_rhet, y_aspect, y_summary, y_citation = transform_to_model_input_multiple(flat_sentences)
    print("Data size: " + str(len(x)))
    return x, y_arg, y_rhet, y_aspect, y_summary, y_citation


def main():
    print("Process started")
    sentences = parse_comp_files(
        r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\compiled_corpus")
    flat_sentences = [item for sublist in sentences for item in sublist]
    #x, y_arg, y_rhet = transform_to_model_input(flat_sentences)
    x, y_arg = transform_to_model_input(flat_sentences)
    print("x", x)
    print("Y", y_arg)
    print("Process ended")


if __name__ == "__main__":
    main()
