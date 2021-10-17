import sys
sys.path.append("./src/")
from util import *
from train_helper import training_loop
from dataset import read_data
from FE import add_features
from config import read_config, update_config, prepare_args

if __name__ == "__main__":
    arg = prepare_args()
    config = read_config(arg.model_config, arg.debug)
    config = update_config(config)
    if config is not None:
        print("Training with ", arg.model_config, " Configuration")
        train, _ = read_data(config)
        print("Read Data: ", train.shape)
        train = add_features(train)
        print("Build Features: ", train.shape)
        seed_torch(seed=config.seed)
        training_loop(train.copy(), config)
