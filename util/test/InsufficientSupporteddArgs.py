import pandas as pd
import tensorflow as tf

from BERT import run_classifier
from BERT import tokenization

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

path = r"C:\Users\Wifo\Documents\Universität Mannheim\Master\Masterthesis\Datasets\Insufficient Arg Support\data-tokenized.tsv"
# doc = codecs.open(path, 'r', encoding='utf-8', errors='ignore')
# print(type(doc))

insufficientSupper_corpora = pd.read_csv(path, delimiter='\t', index_col=None, header=0, encoding='unicode_escape')
print("Colums of the corpus\t")
for col in insufficientSupper_corpora.columns:
    print(col)
insufficientSupper_corpora["ANNOTATION"].fillna("sufficient", inplace=True)
# insufficientSupper_corpora["ANNOTATION"] = (insufficientSupper_corpora["ANNOTATION"] == "sufficient").astype(int) # sufficient := 1
label_list_insufficient = insufficientSupper_corpora.ANNOTATION.unique()
print(type(label_list_insufficient))

# insufficientSupper_corpora["ID"] = pd.Series()
# insufficientSupper_corpora["ID"] = insufficientSupper_corpora["ID"].\
#    apply(lambda x: "{}{}".format ( insufficientSupper_corpora["ESSAY"] , insufficientSupper_corpora["ARGUMENT"]))

print(insufficientSupper_corpora.head())
# print(insufficientSupper_corpora[:3])


print(insufficientSupper_corpora.describe(include='all'))

# print("How many essays:\n", insufficientSupper_corpora.ESSAY.value_counts())
print("Unique values for annotation \n", insufficientSupper_corpora.ANNOTATION.value_counts())
# print("unique values for annotatorid \n", insufficientSupper_corpora.annotatorid.unique())
# print("unique values for id \n", insufficientSupper_corpora.id.unique())
# print("How many unique ones \n", len(insufficientSupper_corpora.loc[: ,"#id"].unique()))


train_InputExamples = insufficientSupper_corpora.apply(
    lambda x: run_classifier.InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this example
                                          text_a=x["TEXT"],
                                          text_b=None,
                                          label=x["ANNOTATION"]), axis=1)

print(len(train_InputExamples))
tokenizerSN = tokenization.FullTokenizer(
    vocab_file=r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\BERT_checkpoint\uncased_L-12_H-768_A-12\vocab.txt",
    do_lower_case=True)

print(tokenizerSN.tokenize("I love you with all my heart"))

# We'll set sequences to be at most 128 tokens long.
MAX_SEQ_LENGTH = 128
# Convert our train and test features to InputFeatures that BERT understands.
# train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
# test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
features = run_classifier.convert_examples_to_features(train_InputExamples, label_list_insufficient, MAX_SEQ_LENGTH,
                                                       tokenizer=tokenizerSN)
print(len(features))

model_fn = run_classifier.model_fn_builder(
    bert_config=r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\BERT_checkpoint\uncased_L-12_H-768_A-12\bert_config.json",
    num_labels=len(label_list_insufficient),
    init_checkpoint=r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\BERT_checkpoint\uncased_L-12_H-768_A-12",
    learning_rate=2e-5,
    num_train_steps=5,
    num_warmup_steps=1,
    use_tpu=False,
    use_one_hot_embeddings=False)

# print(type(model_fn))

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
warmup_proportion = 0.1
tpu_cluster_resolver = None
save_checkpoints_steps = 99999999  # <----- don't want to save any checkpoints
iterations_per_loop = 1000
num_tpu_cores = 4
is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
master = None
output_dir = r"C:\Users\Wifo\PycharmProjects\Masterthesis\data\BERT_checkpoint\uncased_L-12_H-768_A-12"
run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    master=master,
    model_dir=output_dir,
    save_checkpoints_steps=save_checkpoints_steps,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=iterations_per_loop,
        num_shards=num_tpu_cores,
        per_host_input_for_training=is_per_host))

use_tpu = False
estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=use_tpu,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=TRAIN_BATCH_SIZE)

print(type(estimator))

estimator.train(input_fn=features, max_steps=3)
