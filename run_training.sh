#!/usr/bin/env bash

# General paths to data and model folders
# change these to apply locally
python_path="python"
train_script="/home/vincent/Audio_blind_source_separation/main.py"
output_folder="/home/vincent/Audio_blind_source_separation/results/models/"
use_cuda="true" # "true" uses cuda, false does not

mode="train"
model_type="VGG_like_CNN"
data_set_type="DCASE2013_remixed_data_set"

source activate ABSS

model_name="VGG_like_10k_DO_0_0_0_0_0_0"
echo Training VGG_like_10k
${python_path} ${train_script} --mode ${mode} --model_type ${model_type} --classification_mapping FC --data_set_type ${data_set_type} --data_folder Datadir/remixed_DCASE2013_10k --loss_f BCE --use_batch_norm True --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric f1-score --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo

model_name="VGG_like_10k_DO_1_2_3_4_5_0"
echo Training VGG_like_10k
${python_path} ${train_script} --mode ${mode} --model_type ${model_type} --drop_out_probs 0.1 0.2 0.3 0.4 0.5 0.0 --classification_mapping FC --data_set_type ${data_set_type} --data_folder Datadir/remixed_DCASE2013_10k --loss_f BCE --use_batch_norm True --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric f1-score --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo

model_name="VGG_like_10k_DO_2_2_2_2_2_0"
echo Training VGG_like_10k
${python_path} ${train_script} --mode ${mode} --model_type ${model_type} --drop_out_probs 0.2 0.2 0.2 0.2 0.2 0.0 --classification_mapping FC --data_set_type ${data_set_type} --data_folder Datadir/remixed_DCASE2013_10k --loss_f BCE --use_batch_norm True --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric f1-score --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo

model_name="VGG_like_10k_o_M_8_DO_0_0_0_0_0_0"
echo Training VGG_like_10k
${python_path} ${train_script} --mode ${mode} --model_type ${model_type} --classification_mapping FC --data_set_type ${data_set_type} --data_folder Datadir/remixed_DCASE2013_10k_o_M_8 --loss_f BCE --use_batch_norm True --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric f1-score --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo

model_name="VGG_like_10k_o_M_8_DO_1_2_3_4_5_0"
echo Training VGG_like_10k
${python_path} ${train_script} --mode ${mode} --model_type ${model_type} --drop_out_probs 0.1 0.2 0.3 0.4 0.5 0.0 --classification_mapping FC --data_set_type ${data_set_type} --data_folder Datadir/remixed_DCASE2013_10k_o_M_8 --loss_f BCE --use_batch_norm True --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric f1-score --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo

model_name="VGG_like_10k_o_M_8_DO_2_2_2_2_2_0"
echo Training VGG_like_10k
${python_path} ${train_script} --mode ${mode} --model_type ${model_type} --drop_out_probs 0.2 0.2 0.2 0.2 0.2 0.0 --classification_mapping FC --data_set_type ${data_set_type} --data_folder Datadir/remixed_DCASE2013_10k_o_M_8 --loss_f BCE --use_batch_norm True --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric f1-score --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo