import sys
sys.path.append("./src/")

from util import *
from FE import add_features_choice
from dataset import read_data
from infer_helper import get_test_avg, get_cv_score
from config import read_config, update_config, prepare_args

if __name__ == "__main__":
    arg = prepare_args()
    config = read_config(arg.model_config, arg)
    config = update_config(config)
    if config is not None:
        print("Training with ", arg.model_config, " Configuration")
        print("Get CV Score")
        cv = get_cv_score(config)
        _, test = read_data(config)

        print("Read Data: ", test.shape)
        test = add_features_choice(test.copy(), config)
        print("Build Features: ", test.shape)
        seed_torch(seed=config.seed)
        test_avg = get_test_avg(test, config, cv)
