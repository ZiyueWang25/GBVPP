#!/bin/bash

# experiment 1016
CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_do01 --debug 0 --gpu 1
CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config LSTM4_do01 --debug 0 --gpu 1

CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_do03 --debug 0 --gpu 1
CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config LSTM4_do03 --debug 0 --gpu 1

CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM4_do05 --debug 0 --gpu 1
CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config LSTM4_do05 --debug 0 --gpu 1

CUDA_AVAILABLE_DEVICES=1 python3 train.py --model_config LSTM5_do01 --debug 0 --gpu 1
CUDA_AVAILABLE_DEVICES=1 python3 infer.py --model_config LSTM5_do01 --debug 0 --gpu 1
