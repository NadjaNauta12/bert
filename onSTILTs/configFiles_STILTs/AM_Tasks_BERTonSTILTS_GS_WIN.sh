#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export BERT_BASE_DIR='C:\\Users\\Wifo\\PycharmProjects\\Masterthesis\\data\\BERT_checkpoint\\uncased_L-12_H-768_A-12'
export VOCAB_DIR=$BERT_BASE_DIR/vocab.text
export BERT_CONFIG=$BERT_BASE_DIR/bert_config.json
export BERT_onSTILTS_output_dir='C:\\Users\\Wifo\\PycharmProjects\\Masterthesis\\onSTILTs\\models'


for task_name in "InsufficientArgSupport" "ArgRecognition" "ACI_Lauscher" "ACI_Habernal" "ArgQuality"  "ArgZoningI" ; do
    echo $task_name
    case $task_name in
        InsufficientArgSupport) export data_dir='C:\\Users\\Wifo\\PycharmProjects\\Masterthesis\\data\\Insufficient_Arg_Support'
                              echo $data_dir
                              ;;
        ArgRecognition) export data_dir='C:\\Users\\Wifo\\PycharmProjects\\Masterthesis\\data\\Argument_Recognition'
                              echo $data_dir
                              ;;
        ACI_Lauscher) export data_dir='C:\\Users\\Wifo\\PycharmProjects\\Masterthesis\\data\\Argument_Component_Identification_Lauscher'
                              echo $data_dir
                              ;;
        ACI_Habernal) export data_dir='C:\\Users\\Wifo\\PycharmProjects\\Masterthesis\\data\\Argument_Component_Identification_Habernal'
                              echo $data_dir
                              ;;
        ArgQuality) export data_dir='C:\\Users\\Wifo\\PycharmProjects\\Masterthesis\\data\\Argument_Quality'
                              echo $data_dir
                              ;;
        ArgZoningI) export data_dir='C:\\Users\\Wifo\\PycharmProjects\\Masterthesis\\data\\Argument_Zoning'
                              echo $data_dir
                              ;;
        *)  echo "Sorry"
            break;;
    esac

    python C:\\Users\\Wifo\\PycharmProjects\\Masterthesis\\BERT\\run_classifier.py  \
    --task_name=$task_name \
    --do_train=false \
    --do_eval=false \
    #--do_early_stopping=false \
    --data_dir=$data_dir \
    --vocab_file=$VOCAB_DIR \
    --bert_config_file=$BERT_CONFIG \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=16 \          #"[16,32]" \
    --learning_rate="[2e-5, 3e-5]" \
    --num_train_epochs="[3,4]" \
    --output_dir=$BERT_onSTILTS_output_dir/$task_name

done