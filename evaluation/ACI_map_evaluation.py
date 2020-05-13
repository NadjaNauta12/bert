import codecs
import sys


import numpy as np
import tokenization
import os
import platform
from collections import Counter
from util.PathMux import get_path_to_OS, get_path_to_BERT


# lade gold labels
# lade parsed- test results
# map parsed test results with actual number of words
# be careful with tokenization
# shorten parsed test results accordingly


def _truncate_mapping_wp_limit(token_mappedPrediction_per_sentence):
    BERT_path = get_path_to_BERT() + "/vocab.txt"
    tokenizer = tokenization.FullTokenizer(vocab_file=BERT_path, do_lower_case=True)

    for sentence in token_mappedPrediction_per_sentence:
        wp_length = 0
        for i, word_prediction in enumerate(sentence[0]):
            word = word_prediction
            wp_length += len(tokenizer.tokenize(word_prediction))
            if wp_length > 126:        # ACI limit
                print("haha")
                sentence[0] = sentence[0][:i]
                sentence[1] = sentence[1][:i]
                break


def parse_ACI_results(directory, task,  truncate_evaluation):
    BERT_path = get_path_to_BERT() + "/vocab.txt"
    gold = _get_gold_labels(directory, task)
    # print(len(gold))
    predictions = _get_predictions(directory, task)
    # print(len(predictions))

    assert len(gold) == len(predictions)
    if len(gold) == len(predictions):
        tokenizer = tokenization.FullTokenizer(vocab_file=BERT_path, do_lower_case=True)

        token_mappedPrediction_per_sentence = []
        for (idx, sentence) in enumerate(gold):
            # sentence
            token = [token_meta[1] for token_meta in sentence]
            sentence_string = " ".join(token)
            # print(sentence_string)
            # tokenize
            word_pieces = tokenizer.tokenize(sentence_string)
            # print(len(word_pieces))
            # if idx == 131:
            #     print("hey")
            if len(word_pieces) > 126:
                word_cut = 126
            else:
                word_cut = len(word_pieces)


            # shorten predictions
            sequence_128_predictions = predictions[idx]
            #print(sequence_128_predictions[-5:])
            assert len(sequence_128_predictions) == 128
            sequence_wordpiece_predictions = sequence_128_predictions[1:word_cut + 1] # CUT [CLS] and [SEP]
            #print(sequence_wordpiece_predictions[:5])
            #print(sequence_wordpiece_predictions[-5:])
            assert word_cut == len(sequence_wordpiece_predictions)

            # consolidate predictions if token was splitted
            mapped_predictions = _map_consolidate_predictions(token, word_pieces, sequence_wordpiece_predictions,
                                                              tokenizer)

            assert len(mapped_predictions) == len(token)

            #token_mappedPrediction_per_sentence.append(Sentence_Token_Label_Pair(token, mapped_predictions))
            token_mappedPrediction_per_sentence.append([token, mapped_predictions])

        if truncate_evaluation:
            token_mappedPrediction_per_sentence = _truncate_mapping_wp_limit(token_mappedPrediction_per_sentence)
            assert len(token_mappedPrediction_per_sentence[0]) == len(token_mappedPrediction_per_sentence[1])
        path_output = directory + "/" + task + "-mapped_test_results.tsv"
        write_mapped_prediction(path_output, token_mappedPrediction_per_sentence)


def write_mapped_prediction(path_output, mapped_token_label):
    with codecs.open(path_output, "w", "utf-8") as f_out:
        f_out.write("index\ttoken\tlabel\n")
        for idx, mapped_sentences in enumerate(mapped_token_label):
            for i in range(0, len(mapped_sentences[0])):
                # line = str(idx) + "\t" + str(mapped_sentences[0][i]) + "\t" + str(mapped_sentences[1][i]) + "\n"
                f_out.write( str(idx) + "\t" + mapped_sentences[0][i] + "\t" + str(
                        mapped_sentences[1][i]) + "\n")

            f_out.write("\n")
        f_out.close()


def _map_consolidate_predictions(token, word_pieces, wordpiece_predictions, tokenizer):
    if len(token) == len(word_pieces):
        return wordpiece_predictions

    mapped_predictions = []
    idx_token = 0
    idx_wp = 0
    while idx_token < len(token) and idx_wp < len(word_pieces) and idx_wp < 126:
        if token[idx_token].lower() == word_pieces[idx_wp]:
            mapped_predictions.append(wordpiece_predictions[idx_wp])
            idx_wp += 1
            idx_token += 1
        else:
            idx_range = len(tokenizer.tokenize(token[idx_token]))
            delta = 0
            label_container = []
            while delta < idx_range:
                label_container.append(wordpiece_predictions[idx_wp])
                delta += 1
                idx_wp += 1

            if idx_range > 0:
                idx_token += 1
                most_freq_label = Counter(label_container).most_common()[0][0]
                mapped_predictions.append(most_freq_label)
            else:
                # required only for ACI and  Lauscher Corpora
                idx_token += 1
                mapped_predictions.append("Token_Label.OUTSIDE") # TODO-Note - Change to [CLS] >> evaluate differently

    # ACI Stab tokenization too long, such that there are no predictions for input tokens
    while idx_token < len(token):
        mapped_predictions.append("Token_Label.OUTSIDE") # TODO-Note - Change to [CLS] >> evaluate differently
        idx_token += 1


    assert len(mapped_predictions) == len(token)
    return mapped_predictions


def _get_predictions(directory, task):
    sentences = []
    sentence = []
    input_path = directory + "/" + task + "-parsed_test_results.tsv"
    with codecs.open(input_path, "r", "utf-8") as f_in:
        next(f_in)
        for line in f_in.readlines():
            if "SEPARATOR" in line:
                # append sentence
                # req. for ACI Tasks due to sequence
                sentences.append(sentence)
                if len(sentence) > 128:
                    print("meine")
                sentence = []
            else:
                idx, prediction = np.array(line.split("\t"))
                sentence.append(prediction.rstrip())

        f_in.close()

    return sentences


def _get_gold_labels(directory, task):
    sentences = []
    sentence = []
    input_path = directory + "/" + task + "-goldlabels.tsv"
    ## models_onSTILTs\models\ACI_Stab_32_5e-05_3\ACI_Stab-goldlabels.tsv"
    #input_path = get_path_to_OS() + "/models_onSTILTs/models/ACI_Stab_32_5e-05_3/ACI_Stab-goldlabels.tsv"
    with codecs.open(input_path, "r", "utf-8") as f_in:
        next(f_in)
        for line in f_in.readlines():
            if line == "\n":
                # append sentence
                # req. for ACI Tasks due to sequence
                sentences.append(sentence)
                sentence = []
            else:
                idx, word, prediction = np.array(line.split("\t"))
                sentence.append([idx, word, prediction])

        f_in.close()

    return sentences


class Sentence_Token_Label_Pair(object):

    def __init__(self, token, labels):
        self.token_list = token,
        self.label_list = labels


if __name__ == '__main__':
    # parse_ACI_results(directory=get_path_to_OS() + "/models_onSTILTs/models/ACI_Stab_32_5e-05_3",
    #                   task="ACI_Stab")
    parse_ACI_results(directory=get_path_to_OS() + "/models_onSTILTs/models/ACI_Lauscher_32_2e-05_3",
                      task="ACI_Lauscher", truncate_evaluation=False)
