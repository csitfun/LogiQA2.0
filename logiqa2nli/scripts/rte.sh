export DATA_DIR=../DATA
export TASK_NAME=RTE

CUDA_VISIBLE_DEVICES=0,1 python ../run_DATA.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_eval \
    --do_lower_case \
    --data_dir $DATA_DIR/$TASK_NAME \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size=16 \
    --per_gpu_train_batch_size=16   \
    --gradient_accumulation_steps 2\
    --learning_rate 1e-5 \
    --num_train_epochs 10.0 \
    --logging_steps 5000 \
    --save_steps 5000 \
    --output_dir ./tmp/$TASK_NAME/ \
    #--overwrite_output_dir \
