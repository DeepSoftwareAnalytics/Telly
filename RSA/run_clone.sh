
model_type=fine-tuned
current_time=$(date "+%Y%m%d%H%M%S")
task=clone-detection
output_dir=saved_layer_wised_representations_matrix/${task}
train_data_file=../code/dataset/CSN/python/train.jsonl
mkdir -p ${output_dir}/
echo ${output_dir}

loaded_model_path=reproduce_unixcoder/UniXcoder/downstream-tasks/clone-detection/BCB/saved_models/unixcoder/reproduce/20221011024305/checkpoint-best-f1/model.bin
# mkdir -p log
CUDA_VISIBLE_DEVICES=0  python run.py \
--code_length 256 \
--model_type ${model_type} \
--task ${task}  \
--loaded_model_path ${loaded_model_path} \
--train_data_file  ${train_data_file} \
--output_dir ${output_dir} 2>&1 |tee ${output_dir}/log.txt



