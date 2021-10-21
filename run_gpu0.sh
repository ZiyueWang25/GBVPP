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

CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config LSTM4_base_Huber_delta05 --train_folds 0 --debug 0 --gpu 0
CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config LSTM4_base_Huber_delta1 --train_folds 0 --debug 0 --gpu 0
CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config LSTM4_base_Huber_delta2 --train_folds 0 --debug 0 --gpu 0


