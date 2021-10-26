#!/bin/bash

# experiment 1018
# CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config LSTM4_do005 --train_folds 0 --debug 0 --gpu 0
# CUDA_AVAILABLE_DEVICES=0 python3 infer.py --model_config LSTM4_do005 --train_folds 0 --debug 0 --gpu 0

# experiment 1019
# CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config LSTM4_do0 --train_folds 0 --debug 0 --gpu 0
# CUDA_AVAILABLE_DEVICES=0 python3 infer.py --model_config LSTM4_do0 --train_folds 0 --debug 0 --gpu 0
#
# CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config LSTM5_do0 --train_folds 0 --debug 0 --gpu 0
# CUDA_AVAILABLE_DEVICES=0 python3 infer.py --model_config LSTM5_do0 --train_folds 0 --debug 0 --gpu 0

#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config LSTM4_do0_epoch300 --train_folds 3 4 --debug 0 --gpu 0
#CUDA_AVAILABLE_DEVICES=0 python3 infer.py --model_config LSTM4_do0_epoch300 --train_folds 0 1 2 3 4 --debug 0 --gpu 0
#
#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config LSTM4_do0_epoch300_ROP --train_folds 0 1 2 3 4 --debug 0 --gpu 0
#CUDA_AVAILABLE_DEVICES=0 python3 infer.py --model_config LSTM4_do0_epoch300_ROP --train_folds 0 1 2 3 4 --debug 0 --gpu 0

# CUDA_AVAILABLE_DEVICES=0,1 python3 train.py --model_config LSTM4_base_epoch300_ROP_NoUnitVar --train_folds 1 2 3  --debug 0 --gpu 0 1
# CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config LSTM4_base_epoch300_ROP_FC128 --train_folds 1 2 --debug 0 --gpu 0

# check the effect of autocast
# CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config LSTM4_base_epoch300_ROP --train_folds 0 --debug 0 --gpu 0
# check classification
# CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config Base_Cls --train_folds 0 --debug 0 --gpu 0
# CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config Cls_CH_do01 --train_folds 0 --debug 0 --gpu 0
# CUDA_AVAILABLE_DEVICES=0 python3 infer.py --model_config Cls_CH_do01 --train_folds 0 --debug 0 --gpu 0
# CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config Cls_CH_do025 --train_folds 0 --debug 0 --gpu 0
# CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config Cls_CH_do05 --train_folds 0 --debug 0 --gpu 0
# CUDA_AVAILABLE_DEVICES=0,1 python3 train.py --model_config Fork2 --train_folds 0 --debug 0 --gpu 0 1

# 1023
#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config New_base --train_folds 0 --debug 0 --gpu 0
#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config New_base_InPhaseOnly --train_folds 0 --debug 0 --gpu 0
#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config New_base_OutPhase01 --train_folds 0 --debug 0 --gpu 0
#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config New_base_noUnitVar --train_folds 0 --debug 0 --gpu 0
#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config New_base --train_folds 1 2 3 4 --debug 0 --gpu 0

# CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config New_base_fake --train_folds 0 --debug 0 --gpu 0
# CUDA_AVAILABLE_DEVICES=0 python3 infer.py --model_config New_base_fake --train_folds 0 --debug 0 --gpu 0


#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config base_fake --train_folds 0 --debug 0 --gpu 0
#CUDA_AVAILABLE_DEVICES=0 python3 infer.py --model_config base_fake --train_folds 0 --debug 0 --gpu 0
#
#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config base_hb05 --train_folds 0 --debug 0 --gpu 0
#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config base_hb01 --train_folds 0 --debug 0 --gpu 0
#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config base_fc128 --train_folds 0 --debug 0 --gpu 0
#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config base_UnitVar --train_folds 0 --debug 0 --gpu 0
#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config base_cls --train_folds 0 --debug 0 --gpu 0
#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config ch_cls_do01 --train_folds 0 --debug 0 --gpu 0
#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config base_fake --train_folds 1 2 3 4 --debug 0 --gpu 0
# CUDA_AVAILABLE_DEVICES=0 python3 infer.py --model_config base_fake --train_folds 0 1 2 3 4 --debug 0 --gpu 0
# CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config Base --train_folds 0 1 2 --debug 0 --gpu 0
# CUDA_AVAILABLE_DEVICES=0,1 python3 train.py --model_config PulpFiction --train_folds 0 1 --debug 0 --gpu 0 1
# CUDA_AVAILABLE_DEVICES=0,1 python3 infer.py --model_config PulpFiction --train_folds 0 1 --debug 0 --gpu 0 1

# CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config base_better --train_folds 0 1 2 --debug 0 --gpu 0
# CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config base_better_PL --train_folds 3 4 --debug 0 --gpu 0
# CUDA_AVAILABLE_DEVICES=0 python3 infer.py --model_config base_better --train_folds 0 1 2 3 4 --debug 0 --gpu 0
# CUDA_AVAILABLE_DEVICES=0 python3 infer.py --model_config base_better_PL --train_folds 0 1 2 3 4 --debug 0 --gpu 0

#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config base_better_OP01 --train_folds 0 1 2 --debug 0 --gpu 0
#CUDA_AVAILABLE_DEVICES=0 python3 infer.py --model_config base_better_OP01 --train_folds 0 1 2 --debug 0 --gpu 0
#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config base_better_OP01_noRCTogether --train_folds 0 1 2 --debug 0 --gpu 0
#CUDA_AVAILABLE_DEVICES=0 python3 infer.py --model_config base_better_OP01_noRCTogether --train_folds 0 1 2 --debug 0 --gpu 0
