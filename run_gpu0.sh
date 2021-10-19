#!/bin/bash

# experiment 1018
CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config LSTM4_do005 --train_folds 0 --debug 0 --gpu 0
CUDA_AVAILABLE_DEVICES=0 python3 infer.py --model_config LSTM4_do005 --train_folds 0 --debug 0 --gpu 0

CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config LSTM4_do0 --train_folds 0 --debug 0 --gpu 0
CUDA_AVAILABLE_DEVICES=0 python3 infer.py --model_config LSTM4_do0 --train_folds 0 --debug 0 --gpu 0


CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config LSTM5_do0 --train_folds 0 --debug 0 --gpu 0
CUDA_AVAILABLE_DEVICES=0 python3 infer.py --model_config LSTM5_do0 --train_folds 0 --debug 0 --gpu 0

CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config LSTM4_do0_IP --train_folds 0 --debug 0 --gpu 0
CUDA_AVAILABLE_DEVICES=0 python3 infer.py --model_config LSTM4_do0_IP --train_folds 0 --debug 0 --gpu 0

CUDA_AVAILABLE_DEVICES=0 python3 train.py --model_config LSTM5_do0_IP --train_folds 0 --debug 0 --gpu 0
CUDA_AVAILABLE_DEVICES=0 python3 infer.py --model_config LSTM5_do0_IP --train_folds 0 --debug 0 --gpu 0

