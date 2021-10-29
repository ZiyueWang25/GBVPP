import os
import util
from argparse import ArgumentParser
from datetime import datetime

## TODO:
## 1. transformer based model *
## 2. add SWA
## 3. Feature Importance Analysis *
## 4. Error Analysis
## 5. noise in R & C https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/280996 * 
    # check the prediction under diff R & C *
## 6. KNN features


class Base:
    # data
    kaggle_data_folder = "/home/vincent/Kaggle/data/ventilator-pressure-prediction"
    input_folder = "/home/vincent/Kaggle/GBVPP/input/"
    output_folder = "/home/vincent/Kaggle/GBVPP/output/"
    wandb_key_path = "/home/vincent/Kaggle/GBVPP/input/key.txt"
    pressure_unique_path = "/home/vincent/Kaggle/GBVPP/input/pressure_unique.npy"
    fake_pressure_path = "/home/vincent/Kaggle/GBVPP/input/fake_pressure_1368.csv"
    RC_u_in_median_path = "/home/vincent/Kaggle/GBVPP/input/RC_u_in_median.csv"
    RC_u_in_mean_path = "/home/vincent/Kaggle/GBVPP/input/RC_u_in_mean.csv"

    # general
    debug = False
    model_version = "base_version"
    model_module = "BASE" # "CH", "PulpFiction", "transformer"
    PL_folder = None
    seed = 48
    ckpt_folder = None
    use_lr_finder = False


    # preprocessing
    low_q = 25
    high_q = 75
    unit_var = False
    strict_scale = True
    fe_type = "own"

    # features
    use_fake_pressure = False
    use_crossSectional_features = True
    use_RC_together = True
    drop_useless_cols = True

    # Model - LSTM
    hidden = [512, 256, 128, 64]
    lstm_do = 0
    bidirectional = True
    nh = 256
    do_prob = 0
    fc = 50
    use_bn_after_lstm = True

    # Model - transformer
    d_model = 256
    n_head = 8
    do_transformer = 0.1
    dim_forward = 1024
    num_layers = 2

    # training
    do_reg = True
    epochs = 300
    es = 20
    train_folds = [0]
    batch_size = 512
    optimizer = "AdamW"
    lr = 1e-3
    weight_decay = 1e-4
    scheduler = 'ReduceLROnPlateau' #'ReduceLROnPlateau'
    warmup = 20
    factor = 0.5
    patience = 10

    use_in_phase_only = False
    out_phase_weight = None
    loss_fnc = "mae" #huber
    delta = None

    # swa
    use_swa = False

    # logging
    use_wandb = True
    wandb_project = "GBVPP_newCV"
    wandb_post = ""
    wandb_group = "newCV"
    print_num_steps = 100

    # speed
    num_workers = 8
    use_auto_cast = False


class base_no_strict_scale(Base):
    strict_scale = False

class base_fork(Base):
    # https://www.kaggle.com/dienhoa/ventillator-fastai-lb-0-169-no-kfolds-no-blend
    strict_scale = False
    hidden = [512, 256, 128, 128]
    fc = 128
    lr = 2e-3


class LSTM4_base_epoch300_ROP_bn(Base):
    wandb_group = "MakePytorchMatch"
    add_bn_after_lstm = True

class LSTM4_base_epoch300_ROP_bn_2(Base):
    wandb_group = "MakePytorchMatch"

class LSTM4_unitVar(LSTM4_base_epoch300_ROP_bn_2):
    unit_var = True


class LSTM4_base_epoch300_ROP_bn_LSTM5(LSTM4_base_epoch300_ROP_bn_2):
    hidden = [256] * 5

class LSTM5_OP01(LSTM4_base_epoch300_ROP_bn_LSTM5):
    out_phase_weight = 0.1

class LSTM5_OP01_fc128(LSTM4_base_epoch300_ROP_bn_LSTM5):
    out_phase_weight = 0.1
    fc = 128

class LSTM5_OP01_huber025(LSTM4_base_epoch300_ROP_bn_LSTM5):
    out_phase_weight = 0.1
    loss_fnc = "huber"
    delta = 0.25

class LSTM5_OP01_huber025_PL(LSTM5_OP01_huber025):
    PL_folder = "/home/vincent/Kaggle/GBVPP/output/LSTM5_OP01_huber025/"

class LSTM5_OP01_huber025_PL2(LSTM5_OP01_huber025):
    PL_folder = "/home/vincent/Kaggle/GBVPP/output/LSTM5_OP01_huber025_PL/"

class LSTM5_OP01_huber025_PL3(LSTM5_OP01_huber025):
    PL_folder = "/home/vincent/Kaggle/GBVPP/output/LSTM5_OP01_huber025_PL2/"

class LSTM5_OP01_huber025_bn(LSTM5_OP01_huber025):
    PL_folder = None

class LSTM6(LSTM4_base_epoch300_ROP_bn_LSTM5):
    hidden = [256] * 6

class LSTM7(LSTM4_base_epoch300_ROP_bn_LSTM5):
    hidden = [256] * 7

class LSTM8(LSTM4_base_epoch300_ROP_bn_LSTM5):
    hidden = [256] * 8


class LSTM5_CLS_do02(LSTM5_OP01_huber025):
    loss_fnc = "ce"
    do_reg = False
    lstm_do = 0.2

class LSTM5_CLS_do01(LSTM5_CLS_do02):
    lstm_do = 0.1
    

class base_better(Base):
    wandb_group = "betterConfig"
    use_bn_after_lstm = True
    use_RC_together = True
    loss_fnc = "huber"
    delta = 0.1
    fc = 64


class PulpFiction(base_better):
    model_module = "PulpFiction"
    hidden = [768, 512, 384, 256, 128]
    hidden_gru = [384, 256, 128, 64]
    fc = 128
    factor = 0.85
    patience = 7
    es = 21


def update_config(config):
    config.model_output_folder = config.output_folder + config.model_version + "/"
    if config.ckpt_folder:
        config.ckpt_folder = config.model_output_folder
    if config.model_output_folder and not os.path.exists(config.model_output_folder):
        os.makedirs(config.model_output_folder)
    if config.debug:
        config.epochs = 1
        config.train_folds = [0]
    if config.use_wandb:
        if config.wandb_group is None:
            config.wandb_group = config.model_module

    config.device = util.get_device(config)
    print("Model Output Folder:", config.model_output_folder)
    return config


def read_config(name, arg=None):
    assert name in globals(), "name is not in " + str(globals())
    config = globals()[name]
    #current_time = datetime.now().strftime("%m%d%H%M")
    config.model_version = name
    if arg is not None:
        config.gpu = arg.gpu
        config.train_folds = arg.train_folds
        if arg.debug:
            print("---- DEBUG -----")
            config.debug = arg.debug
    return config


def prepare_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', nargs='+', type=int, default=[0, 1],  help='used gpu')
    parser.add_argument('--model_config', type=str, help='configuration name for this run')
    parser.add_argument('--debug', nargs='?', type=int, const=0, help='in debug mode or not')
    parser.add_argument('--train_folds', nargs='+', type=int, default=[0, 1, 2, 3, 4],
                        help='under 5 folds CV setting, what fold we want to train')
    args = parser.parse_args()
    return args
