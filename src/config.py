import os
import util


class BaseConfig:
    # data
    kaggle_data_folder = "/media/vincentwang/Backup/kaggle_data/ventilator-pressure-prediction"
    input_folder = "../input/"
    output_folder = "../output/"

    # general
    debug = False
    model_version = "BasicTest"
    model_module = "Fork"
    PL_folder = None
    seed = 48
    ckpt_folder = None

    # preprocessing
    low_q = 0.05
    high_q = 0.95
    unit_var = True

    # LSTM
    hidden = [512, 256, 128, 64]
    bidirectional = True
    nh = 256
    do_prob = 0.1

    # training
    epochs = 20
    es = 10
    train_folds = [0]
    batch_size = 1024
    lr = 1e-3
    weight_decay = 1e-4
    warmup = 0.1
    scheduler = 'cosineWithWarmUp'

    # swa
    use_swa = False

    # logging
    use_wandb = False
    wandb_project = "GBVPP"
    wandb_key_path = "../input/key.txt"
    wandb_post = ""
    print_num_steps = 100

    # speed
    num_workers = 2
    use_dp = True


def updateConfig(config):
    config.model_output_folder = config.output_folder + config.model_version + "/"
    if config.ckpt_folder:
        config.ckpt_folder = config.model_output_folder
    if config.model_output_folder and not os.path.exists(config.model_output_folder):
        os.makedirs(config.model_output_folder)
    if config.debug:
        config.epochs = 1
        config.train_folds = [0]
    config.device = util.get_device()
    print("Model Output Folder:", config.model_output_folder)
    return config


def read_config(name="base"):
    config_list = {"base": BaseConfig}
    assert name in config_list, "name is not in config_list.keys()" + list(config_list.keys())
    return config_list[name]
