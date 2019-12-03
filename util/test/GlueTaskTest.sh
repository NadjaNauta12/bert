#!/usr/bin/sh
export CUDA_VISIBLE_DEVICES=3
if [ "$OS" = "Windows_NT" ] ; then
  export BERT_BASE_DIR='C:\Users\Wifo\PycharmProjects\Masterthesis\data\BERT_checkpoint\uncased_L-12_H-768_A-12'
  export GLUE_DIR='C:\Users\Wifo\Documents\Universität_Mannheim\Master\Masterthesis\glue_data'
  export output_dir='C:\Users\Wifo\Documents\Universität_Mannheim\Master\Masterthesis\glue_data_output2'

  python  C:\\Users\\Wifo\\PycharmProjects\\Masterthesis\\BERT\\run_classifier.py \
    --task_name=MRPC \
    --do_train=true \
    --do_eval=true \
    --data_dir=$GLUE_DIR/MRPC \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=16 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=/tmp/mrpc_output/
else
    echo "Looks like UNIX to me..."
    export BERT_BASE_DIR="/work/anlausch/uncased_L-12_H-768_A-12"
    export VOCAB_DIR=$BERT_BASE_DIR/vocab.txt
    export BERT_CONFIG=$BERT_BASE_DIR/bert_config.json
    export GLUE_DIR='/work/anlausch/glue_data'
    export output_dir='/work/nseemann/data/glue_data_output'

    python  /work/nseemann/BERT/run_classifier.py \
      --task_name=MRPC \
      --do_train=true \
      --do_eval=true \
      --data_dir=$GLUE_DIR/MRPC \
      --vocab_file=$BERT_BASE_DIR/vocab.txt \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
      --max_seq_length=128 \
      --train_batch_size=32 \
      --learning_rate=2e-5 \
      --num_train_epochs=3.0 \
      --output_dir=$output_dir
fi