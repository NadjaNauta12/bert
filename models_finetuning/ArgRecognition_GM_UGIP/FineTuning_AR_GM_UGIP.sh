#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=2

if [ "$OS" = "Windows_NT" ] ; then
   echo "Windows_NT"
else
    echo "Looks like UNIX to me..."
    export data_dir="/data"
    # # # # # # # # # # change the following three lines
    export BERT_BASE_DIR="/uncased_L-12_H-768_A-12"
    export BERT_finetuning_output_dir="/models_finetuning/ArgRecognition_GM_UGIP"
    export BERT_onSTILTs="/models_onSTILTs/models/ArgRecognition_32_2e-05_4_GM_UGIP/model.ckpt-112"
    # # # # # # # # # #
    export VOCAB_DIR=$BERT_BASE_DIR/vocab.txt
    export BERT_CONFIG=$BERT_BASE_DIR/bert_config.json

    echo ">>>>> TRAINING <<<<<"

    for task_name in   "InsufficientArgSupport"  "ArgQuality" "ArgZoningI" "ACI_Stab" "ACI_Lauscher"  ; do
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
            ACI_Stab) data_dir+="/Argument_Component_Identification_Stab"
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

        python BERT/run_classifier_multipleParameter.py  \
        --task_name=$task_name \
        --do_train=true \
        --do_eval=true \
        --do_predict=true \
        --data_dir=$data_dir \
        --vocab_file=$VOCAB_DIR \
        --bert_config_file=$BERT_CONFIG \
        --init_checkpoint=$BERT_onSTILTs \
        --max_seq_length=128 \
        --train_batch_size="[32]" \
        --learning_rate="[5e-5, 3e-5, 2e-5]" \
        --num_train_epochs="[3, 4]" \
        --output_dir=$BERT_finetuning_output_dir/$task_name

    done
      # AR GM Setting
    for task_name in "ArgRecognition"  ; do
        echo $task_name
        case $task_name in

            ArgRecognition) data_dir+="/Argument_Recognition"
                                  echo $data_dir
                                  ;;
            *)  echo "Something went wrong"
                exit
                ;;
        esac

        python BERT/run_classifier_multipleParameter_GM.py  \
        --task_name=$task_name \
        --do_train=true \
        --do_eval=true \
        --do_predict=true \
        --data_dir=$data_dir \
        --vocab_file=$VOCAB_DIR \
        --bert_config_file=$BERT_CONFIG \
        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
        --max_seq_length=512 \
        --train_batch_size="[8]" \
        --learning_rate="[5e-5, 3e-5, 2e-5]" \
        --num_train_epochs="[3, 4]" \
        --output_dir=$BERT_finetuning_output_dir/$task_name

    done

    echo ">>>>> Training completed <<<<<"


#    echo ">>>>> EVALUATION <<<<<"
#    for task_name in "ArgQuality" "ArgZoningI" "InsufficientArgSupport" "ArgRecognition" "ACI_Stab" "ACI_Lauscher"  ; do
#        python evaluation/parse_predictions.py  \
#        --task=$task_name \
#        --input_path=$BERT_finetuning_output_dir/$task_name \
#        --train_batch_size="[32]" \
#        --learning_rate="[5e-5, 3e-5, 2e-5]" \
#        --num_train_epochs="[3, 4]"
#    done
#    echo ">>>>> Evaluation completed <<<<<"


fi