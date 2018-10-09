#!/usr/bin/env bash

# General paths to data and model folders
# change these to apply locally
python_path="python"
train_script="/home/vincent/Audio_blind_source_separation/main.py"
data_folder="/home/vincent/Audio_blind_source_separation/Datadir/remixed_DCASE2013_2k"
output_folder="/home/vincent/Audio_blind_source_separation/results/models_meeting_28_09_2018/"  # / needed !
use_cuda="true"

model_type="VGG_like_CNN"
data_set_type="DCASE2013_remixed_data_set"

source activate ABSS

# ----------------------------------------------------------------------------------------------------------------------
# ------  GMP ------
model_name="baseline_2k_GMP_dropout_1D_3_3_3_3_3_0"
echo Training ${model_name}
${python_path} ${train_script} --mode train --model_type ${model_type} --classification_mapping GMP --drop_out_probs 0.3 0.3 0.3 0.3 0.3 0.0 --data_set_type ${data_set_type} --data_folder ${data_folder} --loss_f BCE --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo Evaluating AUC per class on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --model_type ${model_type} --data_set_type ${data_set_type} --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="baseline_2k_GMP_dropout_1D_3_3_3_3_3_3"
echo Training ${model_name}
${python_path} ${train_script} --mode train --model_type ${model_type} --classification_mapping GMP --drop_out_probs 0.3 0.3 0.3 0.3 0.3 0.3 --data_set_type ${data_set_type} --data_folder ${data_folder} --loss_f BCE --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo Evaluating AUC on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --model_type ${model_type} --data_set_type ${data_set_type} --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="baseline_2k_GMP_dropout_1D_3_3_3_3_3_5"
echo Training ${model_name}
${python_path} ${train_script} --mode train --model_type ${model_type} --classification_mapping GMP --drop_out_probs 0.3 0.3 0.3 0.3 0.3 0.5 --data_set_type ${data_set_type} --data_folder ${data_folder} --loss_f BCE --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo Evaluating AUC on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --model_type ${model_type} --data_set_type ${data_set_type} --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="baseline_2k_GMP_dropout_1D_3_3_3_3_3_7"
echo Training ${model_name}
${python_path} ${train_script} --mode train --model_type ${model_type} --classification_mapping GMP --drop_out_probs 0.3 0.3 0.3 0.3 0.3 0.7 --data_set_type ${data_set_type} --data_folder ${data_folder} --loss_f BCE --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo Evaluating AUC on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --model_type ${model_type} --data_set_type ${data_set_type} --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="baseline_2k_GMP_dropout_1D_3_3_3_3_3_9"
echo Training ${model_name}
${python_path} ${train_script} --mode train --model_type ${model_type} --classification_mapping GMP --drop_out_probs 0.3 0.3 0.3 0.3 0.3 0.9 --data_set_type ${data_set_type} --data_folder ${data_folder} --loss_f BCE --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo Evaluating AUC on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --model_type ${model_type} --data_set_type ${data_set_type} --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo


# ----------------------------------------------------------------------------------------------------------------------
# ------  GAP ------
model_name="baseline_2k_GAP_dropout_1D_3_3_3_3_3_3"
echo Training ${model_name}
${python_path} ${train_script} --mode train --model_type ${model_type} --classification_mapping GAP --drop_out_probs 0.3 0.3 0.3 0.3 0.3 0.3 --data_set_type ${data_set_type} --data_folder ${data_folder} --loss_f BCE --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo Evaluating AUC on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --model_type ${model_type} --data_set_type ${data_set_type} --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="baseline_2k_GAP_dropout_2D_3_3_3_3_3_0"
echo Training ${model_name}
${python_path} ${train_script} --mode train --model_type ${model_type} --classification_mapping GAP --drop_out_type 2D --drop_out_probs 0.3 0.3 0.3 0.3 0.3 0.0 --data_set_type ${data_set_type} --data_folder ${data_folder} --loss_f BCE --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo Evaluating AUC on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --model_type ${model_type} --data_set_type ${data_set_type} --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo


# ----------------------------------------------------------------------------------------------------------------------
# ------  GWRP ------
model_name="baseline_2k_GWRP_dropout_1D_3_3_3_3_3_3_dc_0"
echo Training ${model_name}
${python_path} ${train_script} --mode train --model_type ${model_type} --classification_mapping GWRP --dc 0.0 --drop_out_probs 0.3 0.3 0.3 0.3 0.3 0.3 --data_set_type ${data_set_type} --data_folder ${data_folder} --loss_f BCE --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo Evaluating AUC on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --model_type ${model_type} --data_set_type ${data_set_type} --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="baseline_2k_GWRP_dropout_1D_3_3_3_3_3_3_dc_1"
echo Training ${model_name}
${python_path} ${train_script} --mode train --model_type ${model_type} --classification_mapping GWRP --dc 0.1 --drop_out_probs 0.3 0.3 0.3 0.3 0.3 0.3 --data_set_type ${data_set_type} --data_folder ${data_folder} --loss_f BCE --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo Evaluating AUC on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --model_type ${model_type} --data_set_type ${data_set_type} --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="baseline_2k_GWRP_dropout_1D_3_3_3_3_3_3_dc_2"
echo Training ${model_name}
${python_path} ${train_script} --mode train --model_type ${model_type} --classification_mapping GWRP --dc 0.2 --drop_out_probs 0.3 0.3 0.3 0.3 0.3 0.3 --data_set_type ${data_set_type} --data_folder ${data_folder} --loss_f BCE --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo Evaluating AUC on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --model_type ${model_type} --data_set_type ${data_set_type} --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="baseline_2k_GWRP_dropout_1D_3_3_3_3_3_3_dc_3"
echo Training ${model_name}
${python_path} ${train_script} --mode train --model_type ${model_type} --classification_mapping GWRP --dc 0.3 --drop_out_probs 0.3 0.3 0.3 0.3 0.3 0.3 --data_set_type ${data_set_type} --data_folder ${data_folder} --loss_f BCE --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo Evaluating AUC on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --model_type ${model_type} --data_set_type ${data_set_type} --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="baseline_2k_GWRP_dropout_1D_3_3_3_3_3_3_dc_4"
echo Training ${model_name}
${python_path} ${train_script} --mode train --model_type ${model_type} --classification_mapping GWRP --dc 0.4 --drop_out_probs 0.3 0.3 0.3 0.3 0.3 0.3 --data_set_type ${data_set_type} --data_folder ${data_folder} --loss_f BCE --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo Evaluating AUC on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --model_type ${model_type} --data_set_type ${data_set_type} --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="baseline_2k_GWRP_dropout_1D_3_3_3_3_3_3_dc_5"
echo Training ${model_name}
${python_path} ${train_script} --mode train --model_type ${model_type} --classification_mapping GWRP --dc 0.5 --drop_out_probs 0.3 0.3 0.3 0.3 0.3 0.3 --data_set_type ${data_set_type} --data_folder ${data_folder} --loss_f BCE --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo Evaluating AUC on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --model_type ${model_type} --data_set_type ${data_set_type} --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="baseline_2k_GWRP_dropout_1D_3_3_3_3_3_3_dc_6"
echo Training ${model_name}
${python_path} ${train_script} --mode train --model_type ${model_type} --classification_mapping GWRP --dc 0.6 --drop_out_probs 0.3 0.3 0.3 0.3 0.3 0.3 --data_set_type ${data_set_type} --data_folder ${data_folder} --loss_f BCE --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo Evaluating AUC on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --model_type ${model_type} --data_set_type ${data_set_type} --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="baseline_2k_GWRP_dropout_1D_3_3_3_3_3_3_dc_7"
echo Training ${model_name}
${python_path} ${train_script} --mode train --model_type ${model_type} --classification_mapping GWRP --dc 0.7 --drop_out_probs 0.3 0.3 0.3 0.3 0.3 0.3 --data_set_type ${data_set_type} --data_folder ${data_folder} --loss_f BCE --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo Evaluating AUC on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --model_type ${model_type} --data_set_type ${data_set_type} --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="baseline_2k_GWRP_dropout_1D_3_3_3_3_3_3_dc_8"
echo Training ${model_name}
${python_path} ${train_script} --mode train --model_type ${model_type} --classification_mapping GWRP --dc 0.8 --drop_out_probs 0.3 0.3 0.3 0.3 0.3 0.3 --data_set_type ${data_set_type} --data_folder ${data_folder} --loss_f BCE --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo Evaluating AUC on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --model_type ${model_type} --data_set_type ${data_set_type} --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="baseline_2k_GWRP_dropout_1D_3_3_3_3_3_3_dc_9"
echo Training ${model_name}
${python_path} ${train_script} --mode train --model_type ${model_type} --classification_mapping GWRP --dc 0.9 --drop_out_probs 0.3 0.3 0.3 0.3 0.3 0.3 --data_set_type ${data_set_type} --data_folder ${data_folder} --loss_f BCE --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo Evaluating AUC on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --model_type ${model_type} --data_set_type ${data_set_type} --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="baseline_2k_GWRP_dropout_1D_3_3_3_3_3_3_dc_10"
echo Training ${model_name}
${python_path} ${train_script} --mode train --model_type ${model_type} --classification_mapping GWRP --dc 1.0 --drop_out_probs 0.3 0.3 0.3 0.3 0.3 0.3 --data_set_type ${data_set_type} --data_folder ${data_folder} --loss_f BCE --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo Evaluating AUC on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --model_type ${model_type} --data_set_type ${data_set_type} --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="baseline_2k_GWRP_dropout_2D_3_3_3_3_3_0_dc_5"
echo Training ${model_name}
${python_path} ${train_script} --mode train --model_type ${model_type} --classification_mapping GWRP --dc 0.5 --drop_out_type 2D --drop_out_probs 0.3 0.3 0.3 0.3 0.3 0.0 --data_set_type ${data_set_type} --data_folder ${data_folder} --loss_f BCE --n_epochs 500 --dev_every 10 --learning_rate 0.0001 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo
echo Training finished
echo Evaluating AUC on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --model_type ${model_type} --data_set_type ${data_set_type} --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo