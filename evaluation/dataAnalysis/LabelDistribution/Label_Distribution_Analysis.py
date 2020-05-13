from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from util.load_datasets import AQ_loader, AZ_loader, ISA_loader, AR_loader, load_conll, load_comp
import numpy as np
import pandas as pd


def _label_distribution_ACI_Lauscher():
    train = load_conll.load_ACI_Lauscher(1)
    dev = load_conll.load_ACI_Lauscher(2)
    test = load_conll.load_ACI_Lauscher(3)
    data = np.concatenate((train, dev, test), axis=0)
    complete_corpora = [item for sublist in data for item in sublist]
    label_list = []
    for ele in complete_corpora:
        labels = [list[1][12:] for list in ele]
        if len(ele) > 126:
            labels = labels[:126]
        label_list.extend(labels)

    label_counts = Counter(label_list)
    # print("ACI - Lauscher - Label Distribution", label_counts)
    # plt.bar(label_counts.keys(), label_counts.values())
    # plt.title("ACI -Lauscher Label Distribution")
    # plt.xticks(rotation='vertical')
    # plt.show()

    df_plot = pd.DataFrame.from_dict(label_counts, orient='index', columns=["Occurrence"]).reset_index()
    df_plot.sort_values(by="Occurrence", ascending=False, inplace=True)
    import seaborn as sns
    sns.set_style('whitegrid', {'grid.linestyle': '--'})
    chart = sns.barplot(y="index", x="Occurrence", data=df_plot, color='darkslategrey')
    chart.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x, p: format(int(x), ',')))
    chart.spines['right'].set_visible(False)
    chart.spines['top'].set_visible(False)
    chart.set_ylabel('Classes')
    chart.set_xlabel('Number of Instances')
    plt.title("BIO Label Distribution $ACI_L$")
    plt.show()



def _label_distribution_ACI_Stab():
    train = load_comp.load_ACI_Stab(1)
    dev = load_comp.load_ACI_Stab(2)
    test = load_comp.load_ACI_Stab(3)
    data = np.concatenate((train, dev, test), axis=0)
    complete_corpora = [item for sublist in data for item in sublist]
    label_list = []
    for ele in complete_corpora:
        labels = [list[1][12:] for list in ele]
        if len(ele) > 126:
            labels = labels[:126]
        label_list.extend(labels)

    label_counts = Counter(label_list)
    plt.bar(label_counts.keys(), label_counts.values())
    plt.title("ACI -Stab Label Distribution")
    plt.xticks(rotation='vertical')
    plt.show()


    label_counts = Counter(label_list)
    df_plot = pd.DataFrame.from_dict(label_counts, orient='index', columns=["Occurrence"]).reset_index()
    df_plot.sort_values(by="Occurrence", ascending=False, inplace=True)

    import seaborn as sns
    sns.set_style('whitegrid', {'grid.linestyle': '--'})
    chart = sns.barplot(y="index", x="Occurrence", data=df_plot, color='darkslategrey')
    chart.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x, p: format(int(x), ',')))
    chart.spines['right'].set_visible(False)
    chart.spines['top'].set_visible(False)
    chart.set_ylabel('Classes')
    chart.set_xlabel('Number of Instances')
    plt.title("BIO Label Distribution $ACI_S$")

    plt.show()

def _label_distribution_ISA():
    train = ISA_loader.get_InsuffientSupport_datset_byFilter(case=1)
    dev = ISA_loader.get_InsuffientSupport_datset_byFilter(case=2)
    test = ISA_loader.get_InsuffientSupport_datset_byFilter(case=3)
    complete_corpora = pd.concat([train, dev, test], ignore_index=True)
    label_counts = Counter(complete_corpora["ANNOTATION"])
    plt.bar(label_counts.keys(), label_counts.values())
    plt.title("ISA Label Distribution")
    plt.show()


def _label_distribution_AQ():
    train = AQ_loader.load_ArgQuality_datset(1)
    dev = AQ_loader.load_ArgQuality_datset(2)
    test = AQ_loader.load_ArgQuality_datset(3)
    complete_corpora = pd.concat([train, dev, test], ignore_index=True)
    label_counts = Counter(complete_corpora["label"])
    plt.bar(label_counts.keys(), label_counts.values())
    plt.title("AQ Label Distribution")
    plt.show()


def _label_distribution_AR_both():
    train = AR_loader.get_ArgRecognition_GM_dataset(1)
    dev = AR_loader.get_ArgRecognition_GM_dataset(2)
    test = AR_loader.get_ArgRecognition_GM_dataset(3)
    complete_corpora = pd.concat([train, dev, test], ignore_index=True)
    label_counts = Counter(complete_corpora["label"])
    plt.bar(label_counts.keys(), label_counts.values())
    plt.title("AR - GM Label Distribution")
    plt.show()

    train, dev, test = AR_loader.get_ArgRecognition_dataset(case_ID=1)
    complete_corpora = pd.concat([train, dev, test], ignore_index=True)
    label_counts = Counter(complete_corpora["label"])
    plt.bar(label_counts.keys(), label_counts.values())
    plt.title("AR - GM_UGIP Label Distribution")
    plt.show()


def _label_distribution_AZ():
    train = AZ_loader.get_ArgZoning_dataset(1)
    dev = AZ_loader.get_ArgZoning_dataset(2)
    test = AZ_loader.get_ArgZoning_dataset(3)
    complete_corpora = pd.concat([train, dev, test], ignore_index=True)
    label_counts = Counter(complete_corpora["AZ_category"])
    print(label_counts)
    plt.bar(label_counts.keys(), label_counts.values())
    plt.title("AZ Label Distribution")
    plt.show()


def main():
    _label_distribution_ACI_Stab()
    #_label_distribution_ACI_Lauscher()


if __name__ == '__main__':
    main()
