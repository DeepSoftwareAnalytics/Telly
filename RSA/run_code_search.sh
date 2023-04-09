
model_type=pre-trained
current_time=$(date "+%Y%m%d%H%M%S")
loaded_model_path=" "
output_dir=saved_layer_wised_representations_matrix/${current_time}
task=pre-trained
train_data_file=../code/dataset/CSN/python/train.jsonl
mkdir -p ${output_dir}/${task}/

# mkdir -p log
 python run.py  \
--code_length 256 \
--model_type ${model_type} \
--task ${task} \
--train_data_file  ${train_data_file} \
--output_dir ${output_dir} 2>&1 |tee ${output_dir}/${task}/log.txt
