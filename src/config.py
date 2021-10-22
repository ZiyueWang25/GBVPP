import os
import util
from argparse import ArgumentParser

## TODO:
## 1. transformer based model
## 2. TabNet
## 3. add SWA
## 4. add more features
## 5. Feature Importance Analysis

class Base:
    # data
    kaggle_data_folder = "/home/vincent/Kaggle/data/ventilator-pressure-prediction"
    input_folder = "/home/vincent/Kaggle/GBVPP/input/"
    output_folder = "/home/vincent/Kaggle/GBVPP/output/"
    wandb_key_path = "/home/vincent/Kaggle/GBVPP/input/key.txt"
    pressure_unique_path = "/home/vincent/Kaggle/GBVPP/input/pressure_unique.npy"


    # general
    debug = False
    model_version = "base_version"
    model_module = "BASE"
    PL_folder = None
    seed = 48
    ckpt_folder = None
    use_lr_finder = False

    # preprocessing
    low_q = 5
    high_q = 95
    unit_var = True
    strict_scale = True
    fe_type = "own"

    # LSTM
    hidden = [512, 256, 128, 64]
    bidirectional = True
    nh = 256
    do_prob = 0.1
    fc = 50

    # training
    do_reg = True
    epochs = 200
    es = 20
    train_folds = [0]
    batch_size = 512
    optimizer = "AdamW"
    lr = 1e-3
    weight_decay = 1e-4
    scheduler = 'cosineWithWarmUp' #'ReduceLROnPlateau'
    warmup = 20
    factor = 0.5
    patience = 10

    use_in_phase_only = False
    loss_fnc = "mae"

    # swa
    ## TODO: add SWA
    use_swa = False

    # logging
    use_wandb = True
    wandb_project = "GBVPP"
    wandb_post = ""
    print_num_steps = 100

    # speed
    num_workers = 8

class Fork(Base):
    model_version = "fork"
    model_module = "BASE"
    low_q = 25
    high_q = 75
    unit_var = False
    strict_scale = False
    fe_type = "fork"
    batch_size = 1024
    optimizer = "Adam"
    hidden = [400, 300, 200, 100]
    use_in_phase_only = False

    scheduler = 'ReduceLROnPlateau'
    factor = 0.5
    patience = 10


class LSTM4_do005(Base):
    model_version = "4LSTM_do005"
    model_module = "CH"
    hidden = [512, 256, 128, 64]
    use_in_phase_only = False
    do_prob = 0.05


class LSTM4_do0(LSTM4_do005):
    model_version = "4LSTM_do0"
    do_prob = 0


class LSTM5_do0(LSTM4_do0):
    model_version = "5LSTM_do0"
    hidden = [1024, 512, 256, 128, 64]


class LSTM4_do0_epoch300(LSTM4_do0):
    model_version = "LSTM4_do0_epoch300"
    epochs = 300


class LSTM4_do0_epoch300_ROP(LSTM4_do0):
    model_version = "LSTM4_do0_epoch300_ROP"
    scheduler = 'ReduceLROnPlateau'
    factor = 0.5
    patience = 10
    epochs = 300


class LSTM4_base_epoch300(LSTM4_do0):
    model_version = "LSTM4_base_epoch300"
    model_module = "BASE"
    epochs = 300


class LSTM4_base_epoch300_ROP(LSTM4_do0):
    # best
    model_version = "LSTM4_base_epoch300_ROP"
    model_module = "BASE"
    scheduler = 'ReduceLROnPlateau'
    factor = 0.5
    patience = 10
    epochs = 300


class LSTM4_do0_IP(LSTM4_do0):
    model_version = "4LSTM_do0_IP"
    use_in_phase_only = True


class LSTM5_do0_IP(LSTM5_do0):
    model_version = "5LSTM_do0_IP"
    use_in_phase_only = True


# huber loss
class LSTM4_base_Huber_delta05(Base):
    # best
    model_version = "LSTM4_base_Huber_delta05"
    model_module = "BASE"
    scheduler = 'ReduceLROnPlateau'
    fc = 128
    factor = 0.5
    patience = 10
    epochs = 300
    # loss
    loss_fnc = "huber"
    delta = 0.5


class LSTM4_base_Huber_delta025(LSTM4_base_Huber_delta05):
    model_version = "LSTM4_base_Huber_delta025"
    delta = .25


class LSTM4_base_Huber_delta1(LSTM4_base_Huber_delta05):
    model_version = "LSTM4_base_Huber_delta1"
    delta = 1


class LSTM4_base_Huber_delta2(LSTM4_base_Huber_delta05):
    model_version = "LSTM4_base_Huber_delta2"
    delta = 2


class LSTM4_base_epoch300_ROP_FC128(LSTM4_base_epoch300_ROP):
    # best
    model_version = "LSTM4_base_epoch300_ROP_FC128"
    fc = 128


class LSTM4_base_epoch300_ROP_NoUnitVar(LSTM4_base_epoch300_ROP):
    # best
    model_version = "LSTM4_base_epoch300_ROP_unitVar"
    unit_var = False

class LSTM4_base_Huber_delta05_PL_fc128(LSTM4_base_Huber_delta05):
    model_version = "LSTM4_base_Huber_delta05_PL_fc128"
    PL_folder = "/home/vincent/Kaggle/GBVPP/output/LSTM4_base_epoch300_ROP"


class Base_Cls(Base):
    model_version = "Base_Cls"
    do_reg = False
    fc = 128
    loss_fnc = "ce"
    scheduler = 'ReduceLROnPlateau'


class Cls_CH_do01(Base_Cls):
    model_module = "CH"
    model_version = "Cls_CH_do01"
    do = 0.1


class Cls_CH_do025(Cls_CH_do01):
    model_version = "Cls_CH_do025"
    do = 0.25


class Cls_CH_do05(Cls_CH_do01):
    model_version = "Cls_CH_do05"
    do = 0.5

def update_config(config):
    config.model_output_folder = config.output_folder + config.model_version + "/"
    if config.ckpt_folder:
        config.ckpt_folder = config.model_output_folder
    if config.model_output_folder and not os.path.exists(config.model_output_folder):
        os.makedirs(config.model_output_folder)
    if config.debug:
        config.epochs = 1
        config.train_folds = [0]

    config.device = util.get_device(config)
    print("Model Output Folder:", config.model_output_folder)
    return config


def read_config(name, arg=None):
    assert name in globals(), "name is not in " + str(globals())
    config = globals()[name]
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
