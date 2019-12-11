#export CUDA_VISIBLE_DEVICES=0
#echo "$OS"

export single_run=true
if [ "$OS" = "Windows_NT" ] ; then
    echo "Looks like WIN to me..."
else
    echo "Looks like UNIX to me..."
    export BERT_BASE_DIR="/content/drive/My Drive/Masterthesis/BERT_checkpoint/uncased_L-12_H-768_A-12"
    export VOCAB_DIR="${BERT_BASE_DIR}/vocab.txt"
    export BERT_CONFIG="${BERT_BASE_DIR}/bert_config.json"
    export BERT_onSTILTS_output_dir="/content/drive/My Drive/Masterthesis/onSTILTs/models"
    export data_dir="/content/drive/My Drive/Masterthesis/data"
  echo $BERT_BASE_DIR
	echo "Running Tasks"
    for task_name in "InsufficientArgSupport"; do # "ArgQuality""ACI_Lauscher""ArgRecognition" "ACI_Habernal""   "ArgZoningI" ; do
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

        python /content/bert/BERT/run_classifier_multipleParameter.py  \
        --task_name=$task_name \
        --do_train=true \
        --do_eval=true \
        --do_predict =false \
        --data_dir="${data_dir}" \
        --vocab_file="${VOCAB_DIR}" \
        --bert_config_file="${BERT_CONFIG}" \
        --init_checkpoint="${BERT_BASE_DIR}/bert_model.ckpt" \
        --max_seq_length=128 \
        --train_batch_size="[16]" \
        --learning_rate="[2e-5]" \
        --num_train_epochs="[3]" \
        --output_dir="${BERT_onSTILTS_output_dir}/${task_name}"

    done
fi