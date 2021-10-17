import sys
sys.path.append("./src/")
from util import *
from train_helper import training_loop
from dataset import read_data
from FE import add_features
from config import read_config, update_config, prepare_args

if __name__ == "__main__":
    arg = prepare_args()
    print(arg.model_config, arg.debug)