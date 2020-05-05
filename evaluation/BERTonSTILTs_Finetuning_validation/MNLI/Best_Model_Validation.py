import codecs
import sys


from util.PathMux import get_path_to_OS


# read file
# save all entries in resprective buckets
# for each bucket select the best model according to the f1 score


def main():
    path = get_path_to_OS() + "/models_finetuning_validation/Consolidated_Modelevaluation_Validation.txt"
    AZI_bucket = []
    AQ_bucket = []
    AR_GM_bucket = []
    AR_GM_UGIP_bucket = []
    ACI_Lauscher_bucket = []
    ACI_Stab_bucket = []
    ISA_bucket = []

    with codecs.open(path, "r", "utf8") as f_in:  # no header
        current_STILT = ""

        for line in f_in.readlines():

            if "Task Combination" in line:
                current_STILT = line.split("\t")[1].rstrip()
            elif "Target model" in line:
                target = line.split("\t")[1].rstrip()
            elif "Setting" in line:
                set = line
            elif "Performance" in line:
                continue
            elif "Recall" in line:
                recall = float(line.split("\t")[1].rstrip())
            elif "Precision" in line:
                precision = float(line.split("\t")[1].rstrip())
            elif "F1 score" in line:
                f1 = float(line.split("\t")[1].rstrip())
            elif "Accuracy" in line:
                acc = float(line.split("\t")[1].rstrip())
            elif "###########" in line:  # blocker
                continue
            elif "\n" in line and not "#" in line:
                if target == "ArgZoningI":
                    AZI_bucket.append(
                        finetuned_model_EVA(stilt=current_STILT, setting=set, recall=recall, precision=precision, f1=f1,
                                            accuracy=acc))
                elif target == "InsufficientArgSupport":
                    ISA_bucket.append(
                        finetuned_model_EVA(stilt=current_STILT, setting=set, recall=recall, precision=precision,
                                            f1=f1, accuracy=acc))
                elif target == "ArgQuality":
                    AQ_bucket.append(
                        finetuned_model_EVA(stilt=current_STILT, setting=set, recall=recall, precision=precision,
                                            f1=f1, accuracy=acc))
                elif target == "ACI_Lauscher":
                    ACI_Lauscher_bucket.append(
                        finetuned_model_EVA(stilt=current_STILT, setting=set, recall=recall, precision=precision,
                                            f1=f1, accuracy=acc))
                elif target == "ACI_Stab":
                    ACI_Stab_bucket.append(
                        finetuned_model_EVA(stilt=current_STILT, setting=set, recall=recall, precision=precision,
                                            f1=f1, accuracy=acc))
                elif target == "ArgRecognition_GM":
                    AR_GM_bucket.append(
                        finetuned_model_EVA(stilt=current_STILT, setting=set, recall=recall, precision=precision,
                                            f1=f1, accuracy=acc))
                elif target == "ArgRecognition_GM_UGIP":
                    AR_GM_UGIP_bucket.append(
                        finetuned_model_EVA(stilt=current_STILT, setting=set, recall=recall, precision=precision,
                                            f1=f1, accuracy=acc))

            else:
                print("FEHLER")

        f_in.close()

    print (len(AQ_bucket))# == print
    print (len(AZI_bucket) )
    print (len(ISA_bucket) )


    # assert (len(AQ_bucket) == 36)
    # assert (len(AZI_bucket) == 36)
    # assert (len(ACI_Stab_bucket) == 36)
    # assert (len(ACI_Lauscher_bucket) == 36)
    # assert (len(AR_GM_UGIP_bucket) == 36)
    # assert (len(ISA_bucket) == 36)
    # assert (len(AR_GM_bucket) == 36)
    # get best model each
    best_model_each = [_get_bet_model(AQ_bucket), _get_bet_model(AZI_bucket), _get_bet_model(ISA_bucket)]
    assert len(best_model_each) == 3
    # save in file
    _write_best_target_models(best_model_each)


def _write_best_target_models(best_model_each):
    path = get_path_to_OS() + "/models_finetuning_validation/Best_Targetmodels.txt"
    with codecs.open(path, "w", "utf8") as f_out:
        for ele in best_model_each:
            f_out.write("Target model:\t" + ele.stilt.split("/")[1] + "\n")
            f_out.write("Stilt:\t" + ele.stilt + "\n")
            f_out.write(ele.setting)
            f_out.write("Recall:\t" + str(ele.recall) + "\n")
            f_out.write("Precision:\t" + str(ele.precision) + "\n")
            f_out.write("F1 score:\t" + str(ele.f1_score) + "\n")
            f_out.write("Accuracy:\t" + str(ele.accuracy) + "\n")
            f_out.write("\n")
        f_out.close()


def _get_bet_model(bucket):
    best_score_idx = -1
    best_f1_score = 0.0
    for i, eva_model in enumerate(bucket):
        if eva_model.f1_score > best_f1_score:
            best_score_idx = i
            best_f1_score = eva_model.f1_score
    return bucket[best_score_idx]


class finetuned_model_EVA():

    def __init__(self, stilt, setting, recall, precision, f1, accuracy):
        self.stilt = stilt
        self.setting = setting
        self.recall = recall
        self.precision = precision
        self.f1_score = f1
        self.accuracy = accuracy


if __name__ == '__main__':
    main()
