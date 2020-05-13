import math
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
import scipy.stats
import scipy.optimize
import scipy.spatial
from sklearn.metrics import f1_score
import codecs
import datetime


def _write_significane_test_result(txt, bootstraps, p_value, delta, result_path, B, sample_size):
    currentDT = datetime.datetime.now()
    task_description = txt.split("-")[0].strip()
    file = result_path + "/" + task_description + "_significanceTestResults_" + currentDT.strftime("%H-%M-%S") + ".txt"
    with codecs.open(file,
                     "w", "utf8") as f_out:
        f_out.write(txt + "\n")
        f_out.write("Bootstraps B\t" + str(B) + "\n")
        f_out.write("Sample_size\t" + str(sample_size) + "\n")
        f_out.write("delta observed\t" + str(delta) + "\n")
        f_out.write("P-value\t" + str(p_value) + "\n")
    f_out.close()

    bootstraps.to_csv(path_or_buf=file, mode='a', header=True, index=True, sep='\t', encoding='utf-8')


def _process_bootstraps(f1_bootstraps_STS, f1_bootstraps_STILT, B, sample_size, delta_observed, description,
                        result_path):
    bootstraps = pd.DataFrame({"bootstrap_f1_STS": f1_bootstraps_STS,
                               "bootstrap_f1_STILT": f1_bootstraps_STILT})

    bootstraps['delta'] = bootstraps.apply(lambda row: row.bootstrap_f1_STILT - row.bootstrap_f1_STS, axis=1)
    bootstraps['teststatistic2delta'] = np.where(bootstraps['delta'] > 2 * delta_observed, 1, 0)  # main focus
    bootstraps['teststatisticsimple'] = np.where(bootstraps['delta'] > 0, 1, 0)
    from collections import Counter
    # print(Counter(bootstraps['teststatisticsimple']))
    # print(Counter(bootstraps['teststatistic2delta']))

    # p_value = bootstraps.teststatisticsimple.sum() / B
    # print("P-Value", p_value)
    p_value = bootstraps.teststatistic2delta.sum() / B
    print("P-Value", p_value)

    _write_significane_test_result(description, bootstraps, p_value, delta_observed, result_path, B, sample_size)


def Bootstrap(gold_file, STS_file, STILT_file, description, result_path, B=50, randomSeed=0):
    sample_size = len(gold_file)
    STS_predictions = pd.read_csv(STS_file, sep="\t")
    STILT_predictions = pd.read_csv(STILT_file, sep="\t")
    gold = pd.read_csv(gold_file, sep="\t")

    assert (len(STS_predictions) == len(gold))
    assert (len(STILT_predictions) == len(gold))

    f1_observed_STS = f1_score(gold.label, STS_predictions.prediction, average="macro")
    f1_observed_STILT = f1_score(gold.label, STILT_predictions.prediction, average="macro")
    delta_observed = f1_observed_STILT - f1_observed_STS
    print("Observed delta:", delta_observed)

    f1_bootstraps_STS = []
    f1_bootstraps_STILT = []

    for i in range(B):
        np.random.seed(i * i + randomSeed)
        bootstrap_STS = []
        bootstrap_STILT = []
        bootstrap_gold = []
        # random ID
        samples = np.random.randint(0, sample_size,
                                    sample_size)  # which samples to add to the subsample with repetitions
        for sample_ID in samples:
            bootstrap_gold.append(gold.iloc[sample_ID]["label"])
            bootstrap_STS.append(STS_predictions.iloc[sample_ID]["prediction"])
            bootstrap_STILT.append(STILT_predictions.iloc[sample_ID]["prediction"])

        f1_STS = f1_score(bootstrap_gold, bootstrap_STS, average="macro")
        f1_STILT = f1_score(bootstrap_gold, bootstrap_STILT, average="macro")
        f1_bootstraps_STS.append(f1_STS)
        f1_bootstraps_STILT.append(f1_STILT)

    # bootstraps = pd.DataFrame({"bootstrap_f1_STS": f1_bootstraps_STS,
    #                            "bootstrap_f1_STILT": f1_bootstraps_STILT})
    #
    # bootstraps['delta'] = bootstraps.apply(lambda row: row.bootstrap_f1_STILT - row.bootstrap_f1_STS, axis=1)
    # bootstraps['teststatistic2delta'] = np.where(bootstraps['delta'] > 2 * delta_observed, 1, 0)   # main focus
    # bootstraps['teststatisticsimple'] = np.where(bootstraps['delta'] > 0, 1, 0)
    # from collections import Counter
    # #print(Counter(bootstraps['teststatisticsimple']))
    # #print(Counter(bootstraps['teststatistic2delta']))
    #
    # #p_value = bootstraps.teststatisticsimple.sum() / B
    # #print("P-Value", p_value)
    # p_value = bootstraps.teststatistic2delta.sum() / B
    # #print("P-Value", p_value)
    #
    # _write_significane_test_result(description, bootstraps, p_value, delta_observed, result_path, B, sample_size)
    _process_bootstraps(f1_bootstraps_STS, f1_bootstraps_STILT, B, sample_size, delta_observed, description,
                        result_path)


def Bootstrap_ACI(gold_file, STS_file, STILT_file, description, result_path, B=50, randomSeed=0):
    sample_size = len(gold_file)
    STS_predictions = pd.read_csv(STS_file, sep="\t")
    STILT_predictions = pd.read_csv(STILT_file, sep="\t")
    gold = pd.read_csv(gold_file, sep="\t")

    assert (len(STS_predictions) == len(gold))
    assert (len(STILT_predictions) == len(gold))

    f1_observed_STS = f1_score(gold.label, STS_predictions.label, average="macro")
    f1_observed_STILT = f1_score(gold.label, STILT_predictions.label, average="macro")
    delta_observed = f1_observed_STILT - f1_observed_STS
    print("Observed delta:", delta_observed)

    f1_bootstraps_STS = []
    f1_bootstraps_STILT = []

    for i in range(B):
        np.random.seed(i * i + randomSeed)
        bootstrap_STS = []
        bootstrap_STILT = []
        bootstrap_gold = []
        # random ID
        samples = np.random.randint(0, sample_size,
                                    sample_size)  # which samples to add to the subsample with repetitions
        for sample_ID in samples:
            bootstrap_gold.append(gold.iloc[sample_ID]["label"])
            bootstrap_STS.append(STS_predictions.iloc[sample_ID]["label"])
            bootstrap_STILT.append(STILT_predictions.iloc[sample_ID]["label"])

        f1_STS = f1_score(bootstrap_gold, bootstrap_STS, average="macro")
        f1_STILT = f1_score(bootstrap_gold, bootstrap_STILT, average="macro")
        f1_bootstraps_STS.append(f1_STS)
        f1_bootstraps_STILT.append(f1_STILT)

    _process_bootstraps(f1_bootstraps_STS, f1_bootstraps_STILT, B, sample_size, delta_observed, description,
                        result_path)


def main():
    entry = "./evaluation/significanceTest/data_test"
    results = "./evagit luation/significanceTest/results"

    descr = "Target Insufficient Arg Support - Best Model ACIL"
    print(descr)
    for i in range(1):
        Bootstrap(gold_file=entry + "/InsufficientArgSupport-goldlabels.tsv",
                  STS_file=entry + "/InsufficientArgSupport-parsed_test_results.tsv",
                  STILT_file=entry + "/ACIL_ISA_best_results.tsv",
                  randomSeed=17,
                  result_path=results,
                  description=descr)

    descr = ("Argument Recognition AR(1) - Intra_Training GM - Best Model ACIS")
    print(descr)
    for i in range(1):
        Bootstrap(gold_file=entry + "/ArgRecognition-1-goldlabels.tsv",
                  STS_file=entry + "/ArgRecognition-1_parsed_test_results.tsv",
                  STILT_file=entry + "/ACIS_AR1_best_results.tsv",
                  randomSeed=17,
                  result_path=results,
                  description=descr)

    descr = ("Argument Recognition (AR2) - Inter_Training GM >> UGIP - Best Model ACIL")
    print(descr)
    for i in range(1):
        Bootstrap(gold_file=entry + "/ArgRecognition-2-goldlabels.tsv",
                  STS_file=entry + "/ArgRecognition-2-parsed_test_results.tsv",
                  STILT_file=entry + "/ACIL_AR2_best_results.tsv",
                  randomSeed=17,
                  result_path=results,
                  description=descr)

    descr = ("Argument Quality Prediction -Best Model ACIS")
    print(descr)
    for i in range(1):
        Bootstrap(gold_file=entry + "/ArgQuality-goldlabels.tsv",
                  STS_file=entry + "/ArgQuality-parsed_test_results.tsv",
                  STILT_file=entry + "/ACIS_AQ_best_results.tsv",
                  randomSeed=17,
                  result_path=results,
                  description=descr)

    descr = ("Argument Zoning - AZI -Best Model AR2")
    print(descr)
    for i in range(1):
        Bootstrap(gold_file=entry + "/ArgZoningI-goldlabels.tsv",
                  STS_file=entry + "/ArgZoningI-parsed_test_results.tsv",
                  STILT_file=entry + "/AR2_AZI_best_results.tsv",
                  randomSeed=17,
                  result_path=results,
                  description=descr)

    descr = ("Argument Component Identification Stab - BIO Scheme - Stab Corpora - Best Model ACI Lauscher")
    print(descr)
    for i in range(1):
        Bootstrap_ACI(gold_file=entry + "/ACI_Stab-goldlabels.tsv",
                      STS_file=entry + "/ACI_Stab-mapped_test_results.tsv",
                      STILT_file=entry + "/ACIL_ACI_Stab_best_results.tsv",
                      randomSeed=17,
                      result_path=results,
                      description=descr)

    descr = ("Argument Component Identification Lauscher - BIO Scheme - Lauscher Corpora - Best Model AR1")
    print(descr)
    for i in range(1):
        Bootstrap_ACI(gold_file=entry + "/ACI_Lauscher-goldlabels.tsv",
                      STS_file=entry + "/ACI_Lauscher-mapped_test_results.tsv",
                      STILT_file=entry + "/AR1_ACI_Lauscher-best_results.tsv",
                      randomSeed=17,
                      result_path=results,
                      description=descr)


if __name__ == "__main__":
    main()
