from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from util.load_datasets import AQ_loader, AZ_loader, ISA_loader, AR_loader, load_conll, load_comp
import numpy as np
import pandas as pd


def main():
    train = load_conll.load_ACI_Lauscher(1)
    dev = load_conll.load_ACI_Lauscher(2)
    data = np.concatenate((train, dev), axis=0)
    complete_corpora = [item for sublist in data for item in sublist]
    label_list = []
    for ele in complete_corpora:
        labels = [list[1][12:] for list in ele]
        if len(ele) > 128:
            labels = labels[:128]
        label_list.extend(labels)

    label_counts = Counter(label_list)
    print("ACI - Lauscher - Label Distribution", label_counts)
    # plt.bar(label_counts.keys(), label_counts.values())
    # plt.title("ACI -Lauscher Label Distribution")
    # plt.xticks(rotation='vertical')
    # plt.show()

    train = load_comp.load_ACI_Stab(1)
    dev = load_comp.load_ACI_Stab(2)
    data = np.concatenate((train, dev), axis=0)
    complete_corpora = [item for sublist in data for item in sublist]
    label_list = []
    for ele in complete_corpora:
        labels = [list[1][12:] for list in ele]
        if len(ele) > 128:
            labels = labels[:128]
        label_list.extend(labels)

    label_counts = Counter(label_list)
    print("ACI - Stab - Label Distribution", label_counts)
    # plt.bar(label_counts.keys(), label_counts.values())
    # plt.title("ACI -Stab Label Distribution")
    # plt.xticks(rotation='vertical')
    # plt.show()

    train = ISA_loader.get_InsuffientSupport_datset_byFilter(case=1)
    dev = ISA_loader.get_InsuffientSupport_datset_byFilter(case=2)
    complete_corpora = pd.concat([train, dev], ignore_index=True)
    label_counts = Counter(complete_corpora["ANNOTATION"])
    print("ISA- Label Distribution", label_counts)
    # plt.bar(label_counts.keys(), label_counts.values())
    # plt.title("ISA Label Distribution")
    # plt.show()

    train = AQ_loader.load_ArgQuality_datset(1)
    dev = AQ_loader.load_ArgQuality_datset(2)
    complete_corpora = pd.concat([train, dev], ignore_index=True)
    label_counts = Counter(complete_corpora["label"])
    print("AQ - Label Distribution", label_counts)
    # plt.bar(label_counts.keys(), label_counts.values())
    # plt.title("AQ Label Distribution")
    # plt.show()

    train = AR_loader.get_ArgRecognition_GM_dataset(1)
    dev = AR_loader.get_ArgRecognition_GM_dataset(2)
    complete_corpora = pd.concat([train, dev], ignore_index=True)
    label_counts = Counter(complete_corpora["label"])
    print("AR - GM Setting - Label Distribution", label_counts)
    # plt.bar(label_counts.keys(), label_counts.values())
    # plt.title("AR - GM Label Distribution")
    # plt.show()

    train, dev, test = AR_loader.get_ArgRecognition_dataset(case_ID=1)
    complete_corpora = pd.concat([train, dev], ignore_index=True)
    label_counts = Counter(complete_corpora["label"])
    print("AR - GM_UGIP Setting - Label Distribution", label_counts)
    # plt.bar(label_counts.keys(), label_counts.values())
    # plt.title("AR - GM_UGIP Label Distribution")
    # plt.show()

    train = AZ_loader.get_ArgZoning_dataset(1)
    dev = AZ_loader.get_ArgZoning_dataset(2)
    complete_corpora = pd.concat([train, dev], ignore_index=True)
    label_counts = Counter(complete_corpora["AZ_category"])
    print("AZ - Label Distribution", label_counts)
    # plt.bar(label_counts.keys(), label_counts.values())
    # plt.title("AZ Label Distribution")
    # plt.show()


if __name__ == '__main__':
    main()
