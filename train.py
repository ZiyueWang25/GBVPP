import sys
sys.path.append("./src/")
from util import *
from train_helper import training_loop
from dataset import read_data
from FE import add_features_choice
from config import read_config, update_config, prepare_args
import os
import gc

if __name__ == "__main__":
    arg = prepare_args()
    print(arg.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in arg.gpu)
    config = read_config(arg.model_config, arg)
    config = update_config(config)
    if config is not None:
        print("Training with ", arg.model_config, " Configuration")
        train, _ = read_data(config)
        print("Read Data: ", train.shape)
        train = add_features_choice(train.copy(), config)
        train, NAlist = reduce_mem_usage(train)
        gc.collect()
        print("Build Features: ", train.shape)
        seed_torch(seed=config.seed)
        training_loop(train.copy(), config)
