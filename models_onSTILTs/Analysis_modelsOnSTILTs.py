import os
import platform


def Analysis_AQ(path):
    # for subdir, dirs, files in os.walk(path):
    #     if "ArgQuality" in subdir:
    #         for file in files:
    #             if '_results.tsv' in file:
    #                 with codecs.open(os.path.join(subdir, file), mode="r", encoding="utf8") as eva_file:
    #                     print(eva_file.readline())
    pass



if __name__ == "__main__":
    if os.name == "nt":
        path_base = "C:/Users/Wifo/PycharmProjects/Masterthesis/models_onSTILTS/models"
    elif platform.release() != "4.9.0-11-amd64":  # GOOGLE COLAB
        print("AQ_Google Colab")
        path_base = "/content/drive/My Drive/Masterthesis/models_onSTILTS/models"  # TODO check this path
    else:
        path_base = "/work/nseemann/models_onSTILTS/models"

    Analysis_AQ(path_base)