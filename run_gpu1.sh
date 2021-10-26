#!/bin/bash

# experiment 1018
# CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_do0_IP --train_folds 1 2 3 4 --debug 0 --gpu 1

# CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_do0_IP --train_folds 0 --debug 0 --gpu 1
# CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config LSTM4_do0_IP --train_folds 0 --debug 0 --gpu 1

# CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM5_do0_IP --train_folds 0 --debug 0 --gpu 1
# CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config LSTM5_do0_IP --train_folds 0 --debug 0 --gpu 1
#
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_base_epoch300 --train_folds 2 3 4 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config LSTM4_base_epoch300 --train_folds 0 1 2 3 4 --debug 0 --gpu 1
#
# CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_base_epoch300_ROP --train_folds 0 --debug 0 --gpu 1
# CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config LSTM4_base_epoch300_ROP --train_folds 0 --debug 0 --gpu 1

# CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_base_Huber_delta025 --train_folds 0 --debug 0 --gpu 1
# CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_base_Huber_delta05 --train_folds 0 --debug 0 --gpu 1

#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_base_Huber_delta05_PL_fc128 --train_folds 0 1 2 3 4 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config LSTM4_base_Huber_delta05_PL_fc128 --train_folds 0 1 2 3 4 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_base_Huber_delta1 --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config LSTM4_base_Huber_delta1 --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_base_Huber_delta2 --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config LSTM4_base_Huber_delta2 --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_base_epoch300_ROP_FC128 --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config LSTM4_base_epoch300_ROP_FC128 --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_base_epoch300_ROP_NoUnitVar --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config LSTM4_base_epoch300_ROP_NoUnitVar --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_base_epoch300_ROP_FC128 --train_folds 3 4 --debug 0 --gpu 1
# CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_base_epoch300_ROP_noAutoCast --train_folds 0 --debug 0 --gpu 1

#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM_base_new_fc128 --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM_base_new_fc128_hb01 --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM_base_new_fc128_hb025 --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM_base_new_fc128_hb05 --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_base_epoch300_ROP_RangerLars --train_folds 0 --debug 0 --gpu 1

#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config Cls_CH_do0 --train_folds 0 --debug 0 --gpu 1


# 1023
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config New_base_hb025 --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config New_CH_do0 --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config New_CH_do01 --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config New_CH_do01_CLS --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config New_base_hb025 --train_folds 1 2 3 4 --debug 0 --gpu 1

#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config New_combined_good --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config New_combined_good --train_folds 1 2 3 4 --debug 0 --gpu 1


#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config Base --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_base_epoch300_ROP_bn --train_folds 0 --debug 0 --gpu 1

#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config base_fork --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config base_no_strict_scale --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config base_IP_only --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config base_OP01 --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config base_hb025 --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config ch_cls_do025 --train_folds 0 --debug 0 --gpu 1

#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config base_better --train_folds 3 4 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config base_better_PL --train_folds 0 1 2 --debug 0 --gpu 1

# CUDA_AVAILABLE_DEVICES=0,1 python3 train.py --model_config base_better_OP01_lossMAE --train_folds 0 --debug 0 --gpu 0 1
#CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config base_better_OP01_lossMAE --train_folds 0 1 2 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config base_better_OP01_WarmUp --train_folds 0 1 2 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config base_better_OP01_WarmUp --train_folds 0 1 2 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=0,1 python3 train.py --model_config base_better2 --train_folds 0 --debug 0 --gpu 0 1


CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config base_transformer --train_folds 0 --debug 0 --gpu 1
CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config base_transformer_DM512 --train_folds 0 --debug 0 --gpu 1
CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config base_transformer_do02 --train_folds 0 --debug 0 --gpu 1
CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config base_transformer_bigger --train_folds 3 4  --debug 0 --gpu 1
