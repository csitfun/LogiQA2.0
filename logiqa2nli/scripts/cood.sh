export DATA_DIR=../DATA
export TASK_NAME=COOD

python ../run_nli.py  --model_type bert  --model_name_or_path bert-base-uncased --task_name $TASK_NAME --do_train --do_eval --do_lower_case --data_dir $DATA_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_eval_batch_size=16 --per_gpu_train_batch_size=16   --gradient_accumulation_steps 2 --logging_steps 1000 --save_steps 1000 --learning_rate 2e-5  --eval_all_checkpoints --num_train_epochs 10.0 --output_dir ./tmp/$TASK_NAME/bert-base/
