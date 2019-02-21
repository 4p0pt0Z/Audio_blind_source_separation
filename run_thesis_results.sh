#!/usr/bin/env bash

# General paths to data and model folders
# change these to apply locally
python_path="python"
train_script="/home/vincent/Audio_blind_source_separation/main.py"
data_folder="/home/vincent/ICASSP2018_joint_separation_classification/packed_features/logmel"
audio_folder="/home/vincent/ICASSP2018_joint_separation_classification/mixed_audio/"
output_folder="/home/vincent/Audio_blind_source_separation/thesis_results/"
use_cuda="true"
gpu_no="0"

mask_model_type="VGGLikeMaskModel"
data_set_type="ICASSP2018JointSeparationClassificationDataSet"

n_epochs="300"
test_every="10"
learning_rate="0.001"

source activate abss


# ----------------------------------------------------------------------------------------------------------------------
# ------  GMP  ------
model_name="GMP"
echo Training ${model_name}
${python_path} ${train_script} --mode train --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr sig --classifier_model_type GlobalMaxPooling2d --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo Evaluating AUC per class on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

# ----------------------------------------------------------------------------------------------------------------------
# ------  GAP  ------
model_name="GAP"
echo Training ${model_name}
${python_path} ${train_script} --mode train --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr sig --classifier_model_type GlobalAvgPooling2d --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

# ----------------------------------------------------------------------------------------------------------------------
# ------  GWRP  ------
model_name="GWRP"
echo Training ${model_name}
${python_path} ${train_script} --mode train --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr sig --classifier_model_type GlobalWeightedRankPooling2d --class_dc 0.999 --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

# ----------------------------------------------------------------------------------------------------------------------
# ------  FC  ------
model_name="FC"
echo Training ${model_name}
${python_path} ${train_script} --mode train --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr sig --classifier_model_type ChannelWiseFC2d --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

# ----------------------------------------------------------------------------------------------------------------------
# ------  FC - sort  ------
model_name="FC-sort"
echo Training ${model_name}
${python_path} ${train_script} --mode train --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr sig --classifier_model_type ChannelWiseFC2d --class_sort True --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

# ----------------------------------------------------------------------------------------------------------------------
# ------  CNN-CWFC  ------
model_name="CNN"
echo Training ${model_name}
${python_path} ${train_script} --mode train --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr sig --classifier_model_type DepthWiseCNNClassifier --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo Evaluating AUC per class on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="CNN_2_layer"
echo Training ${model_name}
${python_path} ${train_script} --mode train --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr sig --classifier_model_type DepthWiseCNNClassifier --class_n_blocks 2 --class_conv_k_f 5 5 --class_conv_k_t 5 5 --class_conv_s_f 3 3 --class_conv_s_t 3 3 --class_conv_p_f 0 0 --class_conv_p_t 0 0 --class_pooling_type avg --class_pool_k_f 1 1 --class_pool_k_t 1 1 --class_pool_s_f 1 1 --class_pool_s_t 1 1 --class_pool_pad_type zero --class_pool_p_f 0 0 --class_pool_p_t 0 0 --class_dropout_probs 0.1 0.1 --class_use_batch_norm False --class_activations lr lr --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo Evaluating AUC per class on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="CNN_3_layer"
echo Training ${model_name}
${python_path} ${train_script} --mode train --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr sig --classifier_model_type DepthWiseCNNClassifier --class_n_blocks 3 --class_conv_k_f 5 5 5 --class_conv_k_t 5 5 5 --class_conv_s_f 3 3 3 --class_conv_s_t 3 3 3 --class_conv_p_f 0 0 0 --class_conv_p_t 0 0 0 --class_pooling_type avg --class_pool_k_f 1 1 1 --class_pool_k_t 1 1 1 --class_pool_s_f 1 1 1 --class_pool_s_t 1 1 1 --class_pool_pad_type zero --class_pool_p_f 0 0 0 --class_pool_p_t 0 0 0 --class_dropout_probs 0.1 0.1 0.1 --class_use_batch_norm False --class_activations lr lr lr --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo Evaluating AUC per class on test set
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo


# ----------------------------------------------------------------------------------------------------------------------
# ------  RNN  ------
model_name="RNN"
echo Training ${model_name}
${python_path} ${train_script} --mode train --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr sig --classifier_model_type ChannelWiseRNNClassifier --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo



# ----------------------------------------------------------------------------------------------------------------------
# ------  L1 loss  ------
model_name="GWRP_L1_0_3"
echo Training ${model_name}
${python_path} ${train_script} --mode train --l1_loss_lambda 0.3 --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr sig --classifier_model_type GlobalWeightedRankPooling2d --class_dc 0.999 --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

## ----------------------------------------------------------------------------------------------------------------------
## ------  Softmax  ------
#model_name="GWRP_softmax"
#echo Training ${model_name}
#${python_path} ${train_script} --mode train --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type GlobalWeightedRankPooling2d --class_dc 0.999 --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
#${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
#echo


# ----------------------------------------------------------------------------------------------------------------------
# ------  PCEN  ------
model_name="GWRP_pcen"
echo Training ${model_name}
${python_path} ${train_script} --mode train --l1_loss_lambda 0.3 --feature_type pcen --pcen_s 0.01 --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr sig --classifier_model_type GlobalWeightedRankPooling2d --class_dc 0.999 --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo



###########  ALL SOFTMAX

# ----------------------------------------------------------------------------------------------------------------------
# ------  GMP  ------
model_name="GMP_softmax"
echo Training ${model_name}
${python_path} ${train_script} --mode train --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type GlobalMaxPooling2d --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

# ----------------------------------------------------------------------------------------------------------------------
# ------  GAP  ------
model_name="GAP_softmax"
echo Training ${model_name}
${python_path} ${train_script} --mode train --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type GlobalAvgPooling2d --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

# ----------------------------------------------------------------------------------------------------------------------
# ------  GWRP  ------
model_name="GWRP_softmax"
echo Training ${model_name}
${python_path} ${train_script} --mode train --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type GlobalWeightedRankPooling2d --class_dc 0.999 --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

# ----------------------------------------------------------------------------------------------------------------------
# ------  FC  ------
model_name="FC_softmax"
echo Training ${model_name}
${python_path} ${train_script} --mode train --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type ChannelWiseFC2d --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

# ----------------------------------------------------------------------------------------------------------------------
# ------  FC - sort  ------
model_name="FC-sort_softmax"
echo Training ${model_name}
${python_path} ${train_script} --mode train --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type ChannelWiseFC2d --class_sort True --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

# ----------------------------------------------------------------------------------------------------------------------
# ------  CNN-CWFC  ------
model_name="CNN_softmax"
echo Training ${model_name}
${python_path} ${train_script} --mode train --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type DepthWiseCNNClassifier --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="CNN_2_layer_softmax"
echo Training ${model_name}
${python_path} ${train_script} --mode train --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type DepthWiseCNNClassifier --class_n_blocks 2 --class_conv_k_f 5 5 --class_conv_k_t 5 5 --class_conv_s_f 3 3 --class_conv_s_t 3 3 --class_conv_p_f 0 0 --class_conv_p_t 0 0 --class_pooling_type avg --class_pool_k_f 1 1 --class_pool_k_t 1 1 --class_pool_s_f 1 1 --class_pool_s_t 1 1 --class_pool_pad_type zero --class_pool_p_f 0 0 --class_pool_p_t 0 0 --class_dropout_probs 0.1 0.1 --class_use_batch_norm False --class_activations lr lr --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="CNN_3_layer_softmax"
echo Training ${model_name}
${python_path} ${train_script} --mode train --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type DepthWiseCNNClassifier --class_n_blocks 3 --class_conv_k_f 5 5 5 --class_conv_k_t 5 5 5 --class_conv_s_f 3 3 3 --class_conv_s_t 3 3 3 --class_conv_p_f 0 0 0 --class_conv_p_t 0 0 0 --class_pooling_type avg --class_pool_k_f 1 1 1 --class_pool_k_t 1 1 1 --class_pool_s_f 1 1 1 --class_pool_s_t 1 1 1 --class_pool_pad_type zero --class_pool_p_f 0 0 0 --class_pool_p_t 0 0 0 --class_dropout_probs 0.1 0.1 0.1 --class_use_batch_norm False --class_activations lr lr lr --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo


# ----------------------------------------------------------------------------------------------------------------------
# ------  RNN  ------
model_name="RNN_softmax"
echo Training ${model_name}
${python_path} ${train_script} --mode train --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type ChannelWiseRNNClassifier --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo




### ALL L1 L0SS softmax
# ----------------------------------------------------------------------------------------------------------------------
# ------  GMP  ------
model_name="GMP_softmax_L1_0_3"
echo Training ${model_name}
${python_path} ${train_script} --mode train --l1_loss_lambda 0.3 --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type GlobalMaxPooling2d --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

# ----------------------------------------------------------------------------------------------------------------------
# ------  GAP  ------
model_name="GAP_softmax_L1_0_3"
echo Training ${model_name}
${python_path} ${train_script} --mode train --l1_loss_lambda 0.3 --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type GlobalAvgPooling2d --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

# ----------------------------------------------------------------------------------------------------------------------
# ------  GWRP  ------
model_name="GWRP_softmax_L1_0_3"
echo Training ${model_name}
${python_path} ${train_script} --mode train --l1_loss_lambda 0.3 --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type GlobalWeightedRankPooling2d --class_dc 0.999 --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

# ----------------------------------------------------------------------------------------------------------------------
# ------  FC  ------
model_name="FC_softmax_L1_0_3"
echo Training ${model_name}
${python_path} ${train_script} --mode train --l1_loss_lambda 0.3 --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type ChannelWiseFC2d --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

# ----------------------------------------------------------------------------------------------------------------------
# ------  FC - sort  ------
model_name="FC-sort_softmax_L1_0_3"
echo Training ${model_name}
${python_path} ${train_script} --mode train --l1_loss_lambda 0.3 --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type ChannelWiseFC2d --class_sort True --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

# ----------------------------------------------------------------------------------------------------------------------
# ------  CNN-CWFC  ------
model_name="CNN_softmax_L1_0_3"
echo Training ${model_name}
${python_path} ${train_script} --mode train --l1_loss_lambda 0.3 --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type DepthWiseCNNClassifier --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="CNN_2_layer_softmax_L1_0_3"
echo Training ${model_name}
${python_path} ${train_script} --mode train --l1_loss_lambda 0.3 --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type DepthWiseCNNClassifier --class_n_blocks 2 --class_conv_k_f 5 5 --class_conv_k_t 5 5 --class_conv_s_f 3 3 --class_conv_s_t 3 3 --class_conv_p_f 0 0 --class_conv_p_t 0 0 --class_pooling_type avg --class_pool_k_f 1 1 --class_pool_k_t 1 1 --class_pool_s_f 1 1 --class_pool_s_t 1 1 --class_pool_pad_type zero --class_pool_p_f 0 0 --class_pool_p_t 0 0 --class_dropout_probs 0.1 0.1 --class_use_batch_norm False --class_activations lr lr --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

model_name="CNN_3_layer_softmax_L1_0_3"
echo Training ${model_name}
${python_path} ${train_script} --mode train --l1_loss_lambda 0.3 --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type DepthWiseCNNClassifier --class_n_blocks 3 --class_conv_k_f 5 5 5 --class_conv_k_t 5 5 5 --class_conv_s_f 3 3 3 --class_conv_s_t 3 3 3 --class_conv_p_f 0 0 0 --class_conv_p_t 0 0 0 --class_pooling_type avg --class_pool_k_f 1 1 1 --class_pool_k_t 1 1 1 --class_pool_s_f 1 1 1 --class_pool_s_t 1 1 1 --class_pool_pad_type zero --class_pool_p_f 0 0 0 --class_pool_p_t 0 0 0 --class_dropout_probs 0.1 0.1 0.1 --class_use_batch_norm False --class_activations lr lr lr --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo


# ----------------------------------------------------------------------------------------------------------------------
# ------  RNN  ------
model_name="RNN_softmax_L1_0_3"
echo Training ${model_name}
${python_path} ${train_script} --mode train --l1_loss_lambda 0.3 --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type ChannelWiseRNNClassifier --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo


# ----------------------------------------------------------------------------------------------------------------------
# ------  PCEN  ------
model_name="GWRP_pcen_softmax_L1_0_3"
echo Training ${model_name}
${python_path} ${train_script} --mode train --l1_loss_lambda 0.3 --feature_type pcen --pcen_s 0.01 --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type GlobalWeightedRankPooling2d --class_dc 0.999 --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo




####  Mask model Number of blocks #####
# ----------------------------------------------------------------------------------------------------------------------
# ------  GWRP  ------
model_name="GWRP_2_blocks_softmax_L1_0_3"
echo Training ${model_name}
${python_path} ${train_script} --mode train --l1_loss_lambda 0.3 --mask_model_type ${mask_model_type} --mask_n_blocks 2 --mask_conv_i_c 1 64 --mask_conv_o_c 64 64 --mask_conv_k_f 5 5 --mask_conv_k_t 5 5 --mask_conv_s_f 1 1 --mask_conv_s_t 1 1 --mask_conv_p_f 2 2 --mask_conv_p_t 2 2 --mask_conv_groups 1 1 --mask_dropout_probs 0.1 0.0 --mask_activations lr softmax --classifier_model_type GlobalWeightedRankPooling2d --class_dc 0.999 --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

# ----------------------------------------------------------------------------------------------------------------------
# ------  GWRP  ------
model_name="GWRP_3_blocks_softmax_L1_0_3"
echo Training ${model_name}
${python_path} ${train_script} --mode train --l1_loss_lambda 0.3 --mask_model_type ${mask_model_type} --mask_n_blocks 3 --mask_conv_i_c 1 64 64 --mask_conv_o_c 64 64 64 --mask_conv_k_f 5 5 5 --mask_conv_k_t 5 5 5 --mask_conv_s_f 1 1 1 --mask_conv_s_t 1 1 1 --mask_conv_p_f 2 2 2 --mask_conv_p_t 2 2 2 --mask_conv_groups 1 1 1 --mask_dropout_probs 0.1 0.1 0.0 --mask_activations lr lr softmax --classifier_model_type GlobalWeightedRankPooling2d --class_dc 0.999 --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

# ----------------------------------------------------------------------------------------------------------------------
# ------  GWRP  ------
model_name="GWRP_4_blocks_softmax_L1_0_3"
echo Training ${model_name}using
${python_path} ${train_script} --mode train --l1_loss_lambda 0.3 --mask_model_type ${mask_model_type} --mask_n_blocks 4 --mask_conv_i_c 1 64 64 64 --mask_conv_o_c 64 64 64 64 --mask_conv_k_f 5 5 5 5 --mask_conv_k_t 5 5 5 5 --mask_conv_s_f 1 1 1 1 --mask_conv_s_t 1 1 1 1 --mask_conv_p_f 2 2 2 2 --mask_conv_p_t 2 2 2 2 --mask_conv_groups 1 1 1 1 --mask_dropout_probs 0.1 0.1 0.1 0.0 --mask_activations lr lr lr softmax --classifier_model_type GlobalWeightedRankPooling2d --class_dc 0.999 --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

# ----------------------------------------------------------------------------------------------------------------------
# ------  GWRP  ------
model_name="GWRP_5_blocks_softmax_L1_0_3"
echo Training ${model_name}
${python_path} ${train_script} --mode train --l1_loss_lambda 0.3 --mask_model_type ${mask_model_type} --mask_n_blocks 5 --mask_conv_i_c 1 64 64 64 64 --mask_conv_o_c 64 64 64 64 64 --mask_conv_k_f 5 5 5 5 5 --mask_conv_k_t 5 5 5 5 5 --mask_conv_s_f 1 1 1 1 1 --mask_conv_s_t 1 1 1 1 1 --mask_conv_p_f 2 2 2 2 2 --mask_conv_p_t 2 2 2 2 2 --mask_conv_groups 1 1 1 1 1 --mask_dropout_probs 0.1 0.1 0.1 0.1 0.0 --mask_activations lr lr lr lr softmax --classifier_model_type GlobalWeightedRankPooling2d --class_dc 0.999 --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo

# ----------------------------------------------------------------------------------------------------------------------
# ------  GWRP  ------
model_name="GWRP_6_blocks_softmax_L1_0_3"
echo Training ${model_name}
${python_path} ${train_script} --mode train --l1_loss_lambda 0.3 --mask_model_type ${mask_model_type} --mask_n_blocks 6 --mask_conv_i_c 1 64 64 64 64 64 --mask_conv_o_c 64 64 64 64 64 64 --mask_conv_k_f 5 5 5 5 5 5 --mask_conv_k_t 5 5 5 5 5 5 --mask_conv_s_f 1 1 1 1 1 1 --mask_conv_s_t 1 1 1 1 1 1 --mask_conv_p_f 2 2 2 2 2 2 --mask_conv_p_t 2 2 2 2 2 2 --mask_conv_groups 1 1 1 1 1 1 --mask_dropout_probs 0.1 0.1 0.1 0.1 0.1 0.0 --mask_activations lr lr lr lr lr softmax --classifier_model_type GlobalWeightedRankPooling2d --class_dc 0.999 --data_set_type ${data_set_type} --data_folder ${data_folder} --audio_folder ${audio_folder} --n_epochs ${n_epochs} --test_every ${test_every} --learning_rate ${learning_rate} --scheduler_type stepLR --scheduler_step_size 50 --scheduler_gamma 0.5 --metric roc_auc_score --average weighted --use_cuda ${use_cuda} --gpu_no ${gpu_no} --save_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
${python_path} ${train_script} --mode evaluate --metric roc_auc_score --average None --checkpoint_path ${output_folder}${model_name}.ckpt >> ${output_folder}${model_name}.txt
echo