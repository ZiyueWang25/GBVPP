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

CUDA_AVAILABLE_DEVICES=0,1 python3 train.py --model_config LSTM4_base_epoch300_ROP_NoUnitVar --train_folds 1 2 3  --debug 0 --gpu 0 1
#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config Base_Cls --train_folds 0 --debug 0 --gpu 0
#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config Cls_CH_do01 --train_folds 0 --debug 0 --gpu 0
#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config Cls_CH_do025 --train_folds 0 --debug 0 --gpu 0
#CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config Cls_CH_do05 --train_folds 0 --debug 0 --gpu 0
