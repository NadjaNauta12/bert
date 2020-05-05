import codecs
import sys


from util.PathMux import get_path_to_OS


def main():
    path = get_path_to_OS() + "/evaluation/BERTonSTILTs_Finetuning_validation/AM/DEV_Scores_Consolidated.txt"
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
                target = current_STILT.split(">>")[1].rstrip()
                print(target)
            elif "Setting" in line:
                set = line
            elif "Performance" in line:
                continue
            elif "eval_accuracy" in line:
                eval_accuracy = float(line.split("=")[1].rstrip())
            elif "eval_loss" in line:
                eval_loss = float(line.split("=")[1].rstrip())
            elif "loss" in line:
                loss = float(line.split("=")[1].rstrip())
            elif "global_step" in line:
                steps = float(line.split("=")[1].rstrip())
            elif "###########" in line:  # blocker
                continue
            elif "\n" in line and not "#" in line:
                if target == "ArgZoningI":
                    AZI_bucket.append(
                        DEV_Scores(stilt=current_STILT, setting=set, eval_accuracy=eval_accuracy, eval_loss=eval_loss, loss=loss,
                                   global_step=steps))
                elif target == "InsufficientArgSupport":
                    ISA_bucket.append(
                        DEV_Scores(stilt=current_STILT, setting=set, eval_accuracy=eval_accuracy, eval_loss=eval_loss,
                                   loss=loss, global_step=steps))
                elif target == "ArgQuality":
                    AQ_bucket.append(
                        DEV_Scores(stilt=current_STILT, setting=set, eval_accuracy=eval_accuracy, eval_loss=eval_loss,
                                   loss=loss, global_step=steps))
                elif target == "ACI_Lauscher":
                    ACI_Lauscher_bucket.append(
                        DEV_Scores(stilt=current_STILT, setting=set, eval_accuracy=eval_accuracy, eval_loss=eval_loss,
                                   loss=loss, global_step=steps))
                elif target == "ACI_Stab":
                    ACI_Stab_bucket.append(
                        DEV_Scores(stilt=current_STILT, setting=set, eval_accuracy=eval_accuracy, eval_loss=eval_loss,
                                   loss=loss, global_step=steps))
                elif target == "ArgRecognition_GM":
                    AR_GM_bucket.append(
                        DEV_Scores(stilt=current_STILT, setting=set, eval_accuracy=eval_accuracy, eval_loss=eval_loss,
                                   loss=loss, global_step=steps))
                elif target == "ArgRecognition_GM_UGIP":
                    AR_GM_UGIP_bucket.append(
                        DEV_Scores(stilt=current_STILT, setting=set, eval_accuracy=eval_accuracy, eval_loss=eval_loss,
                                   loss=loss, global_step=steps))

            else:
                print("Processing incorrect")

        f_in.close()

    assert (len(AQ_bucket) == 36)
    assert (len(AZI_bucket) == 36)
    assert (len(ACI_Stab_bucket) == 36)
    assert (len(ACI_Lauscher_bucket) == 36)
    assert (len(ISA_bucket) == 36)
    assert (len(AR_GM_bucket) == 30)
    assert (len(AR_GM_UGIP_bucket) == 30)

    # get best model each
    best_model_each = [_get_bet_model(AQ_bucket), _get_bet_model(AZI_bucket), _get_bet_model(ISA_bucket),
                       _get_bet_model(ACI_Lauscher_bucket), _get_bet_model(ACI_Stab_bucket),
                       _get_bet_model(AR_GM_UGIP_bucket), _get_bet_model(AR_GM_bucket)]
    assert len(best_model_each) == 7
    # save in file
    _write_best_target_models_Dev_score(best_model_each)


def _write_best_target_models_Dev_score(best_model_each):
    path = get_path_to_OS() + "/evaluation/BERTonSTILTs_Finetuning_validation/AM/Best_DEV_Score_Targetmodels.txt"
    with codecs.open(path, "w", "utf8") as f_out:
        for ele in best_model_each:
            f_out.write("Target model:\t" + ele.stilt.split(">>")[1] + "\n")
            f_out.write("Stilt:\t" + ele.stilt + "\n")
            f_out.write(ele.setting)
            f_out.write("eval_accuracy:\t" + str(ele.eval_acc) + "\n")
            f_out.write("eval_loss:\t" + str(ele.eval_loss) + "\n")
            f_out.write("loss score:\t" + str(ele.loss) + "\n")
            f_out.write("global_step:\t" + str(ele.global_step) + "\n")
            f_out.write("\n")
        f_out.close()


def _get_bet_model(bucket):
    best_score_idx = -1
    best_dev_score = 0.0
    for i, eva_model in enumerate(bucket):
        if eva_model.eval_acc > best_dev_score:
            best_score_idx = i
            best_dev_score = eva_model.eval_acc
    return bucket[best_score_idx]


class DEV_Scores:

    def __init__(self, stilt, setting, eval_accuracy, eval_loss, loss, global_step):
        self.stilt = stilt
        self.setting = setting
        self.eval_acc = eval_accuracy
        self.eval_loss = eval_loss
        self.loss = loss
        self.global_step = global_step


if __name__ == '__main__':
    main()
