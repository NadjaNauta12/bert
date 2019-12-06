from load_datasets import data_loader
import numpy as np
from BERT.run_classifier_multipleParameter import InputExample
import tensorflow as tf
import csv
import pandas as pd
from util.load_datasets import ACI_loader, ACI_loader_Lauscher,data_loader

def ArgQuality_Analysis():
    path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Quality"

    def _get_examples(data_dir, descr):
        if descr == "Train" or descr == "Dev":
            path = data_dir + '/9.1_train_dev'
            c = data_loader.load_QualityPrediction_datset(test_set=False)
        else:
            path = data_dir + '/9.1_test'
            c = data_loader.load_QualityPrediction_datset(test_set=True)

        examples = _convert_To_InputExamples(c, descr)
        return examples

    def _convert_To_InputExamples(df, identifiertxt):
        counter = 1
        counter_Seq = 0
        examples = []
        for idx, row in df.iterrows():
            guid = "%s-%s" % (identifiertxt, counter)
            arg1 = row['a1']
            arg2 = row['a2']
            if (len(arg1.split()) > 128 or len(arg2.split()) > 128):
                counter_Seq += 1
                # raise InputLengthExceeded()
            label = row['target_label']
            examples.append(InputExample(guid=guid, text_a=arg1, text_b=arg2, label=label))
            counter += 1
        counter = 0
        print("Sequence Length too long:", counter_Seq)
        return examples

    train = _get_examples(path, "Train")
    test = _get_examples(path, "Test")

    all_InputExamples = np.concatenate((train, test), axis=0)
    print("all Examples ", len(all_InputExamples))


def ArgZoningI_Analysis():
    path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Zoning"

    def _get_examples(data_dir, descr):
        c = data_loader.get_ArgZoning_dataset(path=data_dir)
        examples = _convert_To_InputExamples(c, descr)
        return examples

    def _convert_To_InputExamples(df, identifiertxt):
        counter = 1
        counter_Seq = 0
        min = 128
        examples = []
        for idx, row in df.iterrows():
            guid = "%s-%s" % (identifiertxt, counter)
            text = row['text']
            label = row['target_label']
            if (len(text.split()) > 128):
                a = len( text.split())
                counter_Seq += 1
            if (len(text.split() )< min):
                min = len(text.split())
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
            counter += 1
        counter = 0
        print("Sequence Length too long:", counter_Seq)
        return examples

    train = _get_examples(path, "Train")
    test = _get_examples(path, "Test")

    all_InputExamples = np.concatenate((train, test), axis=0)
    print("all Examples ", len(all_InputExamples))


def ArgRecognition_Analysis():
    path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Recognition"

    def _get_examples(getGM, descr):
        corpus = data_loader.get_ArgRecognition_dataset()
        if getGM:
            corpus_GM = convert_To_InputExamples(corpus[0], descr)
            return corpus_GM
        else:
            corpus_UGIP = convert_To_InputExamples(corpus[1], descr)
            return corpus_UGIP

    def convert_To_InputExamples( df, identifiertxt):
        counter = 1
        counter_Seq = 0
        min = 128
        examples = []
        for idx, row in df.iterrows():
            guid = "%s-%s" % (identifiertxt, counter)
            comment = row['comment_text']
            argument = row['argument_text']
            if (len(comment.split()) > 128 or len(argument.split()) > 128):
                counter_Seq += 1
            if (len(comment.split()) >min and len(comment.split()) <414):
                min = len(comment.split())
            if min == 414:
                print('Hey')
            label = row['label']
            examples.append(InputExample(guid=guid, text_a=comment, text_b=argument, label=label))
            counter += 1
        counter = 0
        print("Sequence Length too long:", counter_Seq)
        return examples

    GM = _get_examples(getGM=True, descr="Analysis")
    print(len(GM))
    UGIP = _get_examples(getGM=False, descr="Analysis")
    print(len(UGIP))


    all_InputExamples = np.concatenate((train, test), axis=0)
    print("all Examples ", len(all_InputExamples))


def InsufficientArgSupport_Analysis():

    def convert_To_InputExamples(df_Insuff, descr):
        counter = 1
        counter_Seq = 0
        examples = []
        min = 128
        for idx, row in df_Insuff.iterrows():
            if idx == 0:
                continue
            guid = "%s-%s" % (descr, counter)
            text = row['TEXT']
            if (len(text.split()) > 128):
                counter_Seq += 1
                # raise InputLengthExceeded()
            if (len(text.split()) <min ):
                min = len(text.split())

            label = row['ANNOTATION']
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
            counter += 1
        print("Sequence Length too long:", counter_Seq)
        return examples

    def __get_examples(filter_list, descr):
        #print("Listlength", len(filter_list))
        insufficient_corpus = data_loader.get_InsuffientSupport_datset()
        filtered = insufficient_corpus.loc[insufficient_corpus["ESSAY_ID"].isin(filter_list)]
        #print(len(filtered))
        dev_InputExamples = convert_To_InputExamples(filtered, descr)
        return dev_InputExamples

    def read_df_splitting( idx):

        complete_split = _read_tsv(
            "C:/Users/Wifo/PycharmProjects/Masterthesis/data/Insufficient_Arg_Support/data-splitting.tsv")
        # print(type(complete_split))
        # print(len(complete_split))
        # print(complete_split[1])
        # print(len(complete_split[1]))
        complete_split = np.array(complete_split)
        cols = list(range(1, 102))
        # cols = ["Essay_ID"] + cols
        # print(cols)
        # print(complete_split[0: , 1:].shape)
        df_split = pd.DataFrame(data=complete_split[0:, 1:], index=complete_split[0:, 0], columns=cols)
        # print(df_split.shape)
        # print(df_split.head())
        df_split.drop(df_split.columns[len(df_split.columns) - 1], axis=1, inplace=True)
        # print(df_split.head())


        train = []
        test = []
        dev = []
        index = df_split.index.values
        for i in range(0, df_split.shape[0]):
            cur = df_split[idx][i]
            if cur == "TRAIN":
                train.append(index[i].strip())
            elif cur == "DEV":
                dev.append(index[i].strip())
            elif cur == "TEST":
                test.append(index[i].strip())
            else:
                print("something went wrong")
                raise NotImplementedError
        print(len(train))
        print(len(test))
        print(len(dev))
        return [dev, train, test]

    def _read_tsv( input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
    dev_id, train_id, test_id = read_df_splitting(1)
    train = __get_examples(filter_list=train_id, descr="train")
    test = __get_examples(filter_list=test_id, descr="test")
    dev = __get_examples(filter_list=dev_id, descr="dev")
    # print(len(train))
    # print(len(test))
    # print(len(dev))



def ACI_Habernal_Analysis():
    path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal"
    def _get_examples( data_dir, descr):
        if descr == "Train" or descr == "Dev":
            path = data_dir + '/train_dev'
            c = ACI_loader.parse_annotations_Habernal(path=path)
        else:
            path = data_dir + '/test'
            c = ACI_loader.parse_annotations_Habernal(path=path)

        examples = convert_To_InputExamples(c, descr)
        return examples

    def convert_To_InputExamples( df, identifiertxt):
        counter = 1
        counter_Seq = 0
        examples = []

        min = 0
        for idx, row in df.iterrows():
            guid = "%s-%s" % (identifiertxt, counter)
            comment = row['Text']
            label = row['Label']
            if (len(comment.split()) > 128):
                counter_Seq += 1
            if (len(comment.split()) > min):
                min = len(comment.split())
            if len(comment.split()) ==0 and comment != '\n':
                print("very short")
            examples.append(InputExample(guid=guid, text_a=comment, text_b=None, label=label))
            counter += 1
        counter = 0
        return examples

    corpora = _get_examples(data_dir=path, descr="Train")
    corpora2 = _get_examples(data_dir=path, descr="Test")
    all = np.concatenate((corpora, corpora2), axis=1)

def ACI_Lauscher_Analysis():
    def _get_examples(self, data_dir, descr):
        #TODO
        annotations_Lauscher =  ACI_loader_Lauscher.parse_annotations_Lauscher(
            r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Lauscher\compiled_corpus")
        df = pd.DataFrame([annotation.as_dict() for annotation in annotations_Lauscher])
        print(df.head())

        return self._convert_To_InputExamples(df, descr)

    def _convert_To_InputExamples(self, df, identifiertxt):
        counter = 1
        counter_Seq = 0
        examples = []
        for idx, row in df.iterrows():
            guid = "%s-%s" % (identifiertxt, counter)
            sentence = row['Text']
            if (len(sentence.split()) > 128):
                counter_Seq += 1
            label = row['Label']
            examples.append(InputExample(guid=guid, text_a=sentence, text_b=None, label=label))
            counter += 1
        counter = 0
        return examples

    corpora = _get_examples(data_dir=data_dir, descr="Train")

if __name__ == "__main__":
    # ArgQuality_Analysis()
    # ArgZoningI_Analysis()
    # ArgRecognition_Analysis()
    InsufficientArgSupport_Analysis()
    # not affected
    # ACI_Habernal_Analysis()