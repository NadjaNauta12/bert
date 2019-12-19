import sklearn
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
import os
import codecs


def _write_eva_results(path, accuracy, f1_score, recall, precision, task, config, setting_AR):
    # path ist ueberdir where models are located
    container_file = "Evaluation_across_" + task + ".tsv"
    if task == "ArgRecognition":
        container_file = "Evaluation_across_" + task + "_" + setting_AR + ".tsv"
    with open(path + "/" + container_file, "a") as myfile:
        myfile.write("\n")
        myfile.write("Setting: \t Size: %s \t Learningrate: %s \t Epochs: %s \n" % config)
        myfile.write('Accuracy:\t %s \n' % accuracy)
        myfile.write('F1 score:\t %s \n' % f1_score)
        myfile.write('Recall:\t %s \n' % recall)
        myfile.write('Precision:\t %s \n' % precision)
        myfile.write("\n")
        myfile.close()


def run_evaluation(path):
    # walk trough all dirs
    # remeber label and prediction
    goldlabel = []
    prediction = []
    for subdir, dirs, files in os.walk(path):
        for dir in dirs:
            print(dir)
            path = subdir + "/" + dir
            for subdir_model, dirs_model, files_model in os.walk(path):

                for file in files_model:
                    if '-goldlabels.tsv' in file:
                        with codecs.open(os.path.join(path, file), mode="r", encoding="utf8") as gold_file:
                            next(gold_file, None)
                            for line in gold_file:
                                goldlabel.append(line.split("\t")[1].rstrip('\n'))
                    elif '-parsed_test_results.tsv' in file:
                        with codecs.open(os.path.join(path, file), mode="r", encoding="utf8") as pred_file:
                            next(pred_file, None)
                            for ln in pred_file:
                                prediction.append(ln.split("\t")[1].rstrip('\n'))
                # files should be ready
                if len(prediction) > 0 and len(prediction) == len(goldlabel):
                    if "ArgRecognition" in dir:
                        task, batch, eta, epochs, setting_AR = dir.split("_")
                    else:
                        task, batch, eta, epochs = dir.split("_")
                        setting_AR = ""
                    # make eva
                    # from sklearn.metrics import precision_recall_fscore_support as score
                    # precision, recall, fscore, support = score(goldlabel, prediction, pos_label='micro')
                    # print('precision: {}'.format(precision))
                    # print('recall: {}'.format(recall))
                    # print('fscore: {}'.format(fscore))
                    # print('support: {}'.format(support))
                    # print()
                    # print()

                    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
                        classification_report, \
                        confusion_matrix
                    print("MACRO")
                    print(f1_score(goldlabel, prediction, average="macro"))
                    print(precision_score(goldlabel, prediction, average="macro"))
                    print(recall_score(goldlabel, prediction, average="macro"))
                    print()
                    print("MICRO")
                    accuracy = accuracy_score(goldlabel, prediction)
                    f1 = f1_score(goldlabel, prediction, average="micro")
                    recall = recall_score(goldlabel, prediction, average="micro")
                    precision = precision_score(goldlabel, prediction, average="micro")
                    classification = classification_report(goldlabel, prediction, digits=3)

                    print('Accuracy:', accuracy)
                    print('F1 score:', f1)
                    print('Recall:', recall)
                    print('Precision:', precision)
                    print('\n classification report:\n')
                    print(classification)
                    from nltk import ConfusionMatrix
                    print(ConfusionMatrix(list(goldlabel), list(prediction)))
                    # write into one file -> with config string and task
                    _write_eva_results(subdir, accuracy, f1, recall, precision, task, config=(batch, eta, epochs),
                                       setting_AR=setting_AR)

                goldlabel = []
                prediction = []
            # ende


def run_evaluation_heatmap(path):
    # walk trough all dirs
    # remeber label and prediction
    goldlabel = []
    prediction = []
    for subdir, dirs, files in os.walk(path):
        for dir in dirs:
            path = subdir + "/" + dir
            for subdir_model, dirs_model, files_model in os.walk(path):

                for file in files_model:
                    if '-goldlabels.tsv' in file:
                        with codecs.open(os.path.join(path, file), mode="r", encoding="utf8") as gold_file:
                            next(gold_file, None)
                            for line in gold_file:
                                goldlabel.append(line.split("\t")[1].rstrip('\n'))
                    elif '-parsed_test_results.tsv' in file:
                        with codecs.open(os.path.join(path, file), mode="r", encoding="utf8") as pred_file:
                            next(pred_file, None)
                            for ln in pred_file:
                                prediction.append(ln.split("\t")[1].rstrip('\n'))

    if prediction is not None and goldlabel is not None:
        if len(prediction) == len(goldlabel):
            # make eva
            pass
            # import seaborn as sns
            # import matplotlib.pyplot as plt
            # from sklearn.metrics import confusion_matrix
            #
            # labels = ['1','2','3','4','5']
            # cm = confusion_matrix(goldlabel, prediction, labels)
            # ax = plt.subplot()
            # sns.heatmap(cm, annot=True, ax=ax);  # annot=True to annotate cells
            #
            # # labels, title and ticks
            # ax.set_xlabel('Predicted labels');
            # ax.set_ylabel('True labels');
            # ax.set_title('Confusion Matrix');
            # ax.xaxis.set_ticklabels(['1','2','3','4','5']);
            # ax.yaxis.set_ticklabels(['1','2','3','4','5'])


def main():
    eva_onSTILTs = True
    eva_finetuning = False
    if eva_onSTILTs:
        if os.name == "nt":
            path = "C:/Users/Wifo/PycharmProjects/Masterthesis/models_onSTILTs/models"
        elif platform.release() != "4.9.0-11-amd64":  # GOOGLE COLAB
            print("AQ_Google Colab")
            path = "/content/drive/My Drive/Masterthesis/models_onSTILTs/models"
        else:
            path = "/work/nseemann/models_onSTILTs/models"

        run_evaluation(path)

    if eva_finetuning:
        pass


if __name__ == "__main__":
    main()
