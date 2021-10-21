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
CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_base_epoch300_ROP --train_folds 0 --debug 0 --gpu 1
CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config LSTM4_base_epoch300_ROP --train_folds 0 --debug 0 --gpu 1


#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_base_Huber_delta025 --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_base_Huber_delta05 --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_base_Huber_delta1 --train_folds 0 --debug 0 --gpu 1
#CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_base_Huber_delta2 --train_folds 0 --debug 0 --gpu 1
