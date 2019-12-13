import os
from random import shuffle, Random
from math import floor
import os.path
import shutil


def _get_file_list_from_dir(datadir, data_type):
    all_files = os.listdir(os.path.abspath(datadir))
    # data_files = list(filter(lambda file: file.endswith('.data'), all_files))
    data_files = list(filter(lambda file: file.endswith(data_type), all_files))
    return data_files


def randomize_files(file_list):
    Random(4).shuffle(file_list)


def get_training_and_testing_sets(file_list, split_train=0.49, split_test=0.7):
    split_index1 = floor(len(file_list) * split_train)
    split_index2 = floor(len(file_list) * split_test)
    training = file_list[:split_index1]
    dev = file_list[split_index1:split_index2]
    testing = file_list[split_index2:]
    return training, dev, testing


def get_two_sets(file_list, split_train=0.7):
    split_index = floor(len(file_list) * split_train)
    uno = file_list[:split_index]
    dos = file_list[split_index:]
    return uno, dos


def split_ACI_Habernal():
    path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal"
    train = []
    test = []
    with open(path + "\\train-test-split.csv", 'r') as f:
        for line in f:
            id, set = line.split(";")
            id = id.replace("\"", "")
            set = set.replace("\"", "").replace("\n", "")
            if set == "TRAIN" and set != "SET":
                train.append(id)
            elif set != "SET":
                test.append(id)
    print("#Test files", len(train))
    print("# Train files", len(test))
    # for f in train:
    #     path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\brat-project-final"
    #     shutil.move(
    #         path + "\\" + f + ".txt"
    #         , r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\train_dev")
    #     '**annotationf file too'
    #     shutil.move(
    #         path + "\\" + f + ".ann"
    #         , r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\train_dev")
    for f in train:
        path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\compiled_corpus"
        shutil.move(
            path + "\\" + f + ".comp"
            , r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\train_dev")
    for f in test:
        path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\compiled_corpus"
        shutil.move(
            path + "\\" + f + ".comp"
            ,
            r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\test")


#
# def split_ACI_Habernal_wrong():
#     list_files = _get_file_list_from_dir(
#         r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\brat-project-final",
#         '.txt')
#     randomize_files(list_files)
#     train, test = get_two_sets(list_files)
#     print("Train")
#     print(train)
#     print("Test")
#     print(test)
#
#     eva_file = open(
#         r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\ACI_Habelernal_Split.txt",
#         "w")
#     eva_file.write(" >>  ACI Habernal Split")
#     eva_file.write("\n")
#     eva_file.write("Train & Dev")
#     eva_file.write("\n")
#
#     eva_file.write(''.join(e + "\n" for e in train))
#     eva_file.write("\n")
#     eva_file.write("\n")
#     eva_file.write("\n")
#     eva_file.write("Test")
#     eva_file.write("\n")
#     eva_file.write(''.join(e + "\n" for e in test))
#     eva_file.close()
#
#     '*** move files'
#
#     for f in train:
#         path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\brat-project-final"
#         shutil.move(
#             path + "\\" + f
#             , r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\train_dev")
#         '**annotationf file too'
#         annfile = f[:len(f) - 4] + ".ann"
#         shutil.move(
#             path + "\\" + annfile
#             , r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\train_dev")

def split_ACI_Habeneral_Train_Dev():
    list_files = _get_file_list_from_dir(
        r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\train_dev",
        '.comp')
    randomize_files(list_files)
    train, dev = get_two_sets(list_files)
    assert len(train) == 225
    assert len(dev) == 97
    print("#Test files", len(train))
    print("# Train files", len(dev))

    for f in train:
        path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\train_dev"
        shutil.move(path + "\\" + f,
                    r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\train")
    for f in dev:
        path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\train_dev"
        shutil.move(path + "\\" + f,
                    r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Component_Identification_Habernal\dev")


def split_AZI():
    file_list = _get_file_list_from_dir(r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Zoning", "az-scixml")
    randomize_files(file_list)
    train, dev, test = get_training_and_testing_sets(file_list, split_train=0.49, split_test=0.7)
    print("Train")
    #print(train)
    print(len(train))
    print("Test")
    #print(test)
    print(len(test))
    print("Dev")
    #print(dev)
    print(len(dev))
    '*** move files'

    for f in train:
        path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Zoning"
        shutil.move(
            path + "\\" + f
            , r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Zoning\train")

    for f in test:
        path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Zoning"
        shutil.move(
            path + "\\" + f
            , r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Zoning\test")

    for f in dev:
        path = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Zoning"
        shutil.move(
            path + "\\" + f
            , r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\Argument_Zoning\dev")


if __name__ == "__main__":
    #split_ACI_Habernal()
    #split_ACI_Habeneral_Train_Dev()
    # split_AZI()
    pass