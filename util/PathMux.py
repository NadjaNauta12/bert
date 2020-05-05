import os
import platform


def get_path_to_OS():
    if os.name == "nt":
        path_base = ""  # TODO PATH

    elif platform.release() != "4.9.0-11-amd64":  # GOOGLE COLAB
        print("AQ_Google Colab")
        path_base = "/content/drive/My Drive/"

    else:
        path_base = ""  # TODO PATH

    return path_base


def get_path_to_BERT():
    if os.name == "nt":
        path_base = ""  # TODO PATH   data/BERT_checkpoint/uncased_L-12_H-768_A-12"

    elif platform.release() != "4.9.0-11-amd64":  # GOOGLE COLAB
        print("AQ_Google Colab")
        path_base = ""  # TODO PATH  data/BERT_checkpoint/uncased_L-12_H-768_A-12"

    else:
        path_base = ""  # TODO PATH

    return path_base
