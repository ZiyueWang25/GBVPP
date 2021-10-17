#!/bin/bash

# experiment 1016
python3 train.py --model_config LSTM4_do01 --debug 0
python3 infer.py --model_config LSTM4_do01 --debug 0
python3 train.py --model_config LSTM4_do02 --debug 0
python3 infer.py --model_config LSTM4_do02 --debug 0
