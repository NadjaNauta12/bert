#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=0
#echo "$OS"

export single_run=true
if [ "$OS" = "Windows_NT" ] ; then
    export BERT_BASE_DIR="C:/Users/Wifo/PycharmProjects/Masterthesis/data/BERT_checkpoint/uncased_L-12_H-768_A-12"
    export VOCAB_DIR=$BERT_BASE_DIR/vocab.text
    export BERT_CONFIG=$BERT_BASE_DIR/bert_config.json
    export BERT_onSTILTS_output_dir="C:/Users/Wifo/PycharmProjects/Masterthesis/onSTILTs/models"
    export data_dir="C:/Users/Wifo/PycharmProjects/Masterthesis/data"

else
    echo "Looks like UNIX to me..."  # TODO change paths for server
    export BERT_BASE_DIR="C:/Users/Wifo/PycharmProjects/Masterthesis/data/BERT_checkpoint/uncased_L-12_H-768_A-12"
    export VOCAB_DIR=$BERT_BASE_DIR/vocab.text
    export BERT_CONFIG=$BERT_BASE_DIR/bert_config.json
    export BERT_onSTILTS_output_dir="C:/Users/Wifo/PycharmProjects/Masterthesis/onSTILTs/models"
    export data_dir="work/Seemann/"
fi

for task_name in "InsufficientArgSupport" ; do # "ArgRecognition" "ACI_Lauscher" "ACI_Habernal" "ArgQuality"  "ArgZoningI" ; do
    echo $task_name
    case $task_name in
        InsufficientArgSupport) data_dir+="/Insufficient_Arg_Support"
                              echo $data_dir
                              ;;
        ArgRecognition) data_dir+="/Argument_Recognition"
                              echo $data_dir
                              ;;
        ACI_Lauscher) data_dir+="/Argument_Component_Identification_Lauscher"
                              echo $data_dir
                              ;;
        ACI_Habernal) data_dir+="/Argument_Component_Identification_Habernal"
                              echo $data_dir
                              ;;
        ArgQuality) data_dir+="/Argument_Quality"
                              echo $data_dir
                              ;;
        ArgZoningI) data_dir+="/Argument_Zoning"
                              echo $data_dir
                              ;;
        *)  echo "Something went wrong"
            exit
            ;;
    esac

    python C:/Users/Wifo/PycharmProjects/Masterthesis/BERT/run_classifier.py  \
    --task_name=$task_name \
    --do_train=false \
    --do_eval=false \
    --do_predict =false \
    --data_dir=$data_dir \
    --vocab_file=$VOCAB_DIR \
    --bert_config_file=$BERT_CONFIG \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size="[16]" \
    --learning_rate="[2e-5]" \
    --num_train_epochs="[3]" \
    --output_dir=$BERT_onSTILTS_output_dir/$task_name

#    --train_batch_size="[16, 32]" \
#    --learning_rate="[2e-5, 3e-5]" \
#    --num_train_epochs="[3, 4]" \
#    --output_dir=$BERT_onSTILTS_output_dir/$task_name
# *** work with lists - TODO switch to lists on server
done