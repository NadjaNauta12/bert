#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=0

export single_run=true
if [ "$OS" = "Windows_NT" ]; then
  echo ">>>>> NOT RUNNING <<<<<"
else
  echo "Looks like UNIX to me..."
  export BERT_BASE_DIR="/uncased_L-12_H-768_A-12"
  export VOCAB_DIR=$BERT_BASE_DIR/vocab.txt
  export BERT_CONFIG=$BERT_BASE_DIR/bert_config.json
  export BERT_onSTILTS_output_dir="models_onSTILTs/models"
  export data_dir="/data"

  echo ">>>>> TRAINING <<<<<"

  python  BERT/MNLI_run_classifier.py \
  --task_name=MNLI \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$GLUE_DIR/MNLI \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size="[32]" \
  --learning_rate="[5e-5, 3e-5, 2e-5]" \
  --num_train_epochs="[3, 4]" \
  --output_dir=$BERT_onSTILTS_output_dir/MNLI


  echo ">>>>> Training completed <<<<<"

  echo ">>>>> EVALUATION <<<<<"
  python evaluation/parse_predictions.py  \
  --task=MNLI \
  --input_path=$BERT_onSTILTS_output_dir/MNLI \
  --train_batch_size="[32]" \
  --learning_rate="[5e-5, 3e-5, 2e-5]" \
  --num_train_epochs="[3, 4]"
  echo ">>>>> Evaluation completed <<<<<"
fi
