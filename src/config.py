import os
import util
from argparse import ArgumentParser
from datetime import datetime

## TODO:
## 1. transformer based model
## 2. TabNet
## 3. add SWA
## 4. Feature Importance Analysis *
## 5. Error Analysis
## 6. noise in R & C https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/280996
    # check the prediction under diff R & C
## 7. KNN features


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
    use_RC_together = False
    drop_useless_cols = False

    # Model - LSTM
    hidden = [512, 256, 128, 64]
    bidirectional = True
    nh = 256
    do_prob = 0
    fc = 50
    use_bn_after_lstm = False

    # Model - transformer
    d_model = 256
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


class base_IP_only(Base):
    use_in_phase_only = True


class base_OP01(Base):
    out_phase_weight = 0.1

class base_hb025(Base):
    loss_fnc = "huber"
    delta = 0.25


class base_hb05(base_hb025):
    delta = 0.5


class base_hb01(base_hb025):
    delta = 0.1


class base_fc128(Base):
    fc = 128

class base_UnitVar(Base):
    unit_var = True

class base_cls(Base):
    model_version = "Base_Cls"
    do_reg = False
    fc = 128
    loss_fnc = "ce"


class ch_cls_do01(base_cls):
    model_module = "CH"
    do = 0.1


class ch_cls_do025(ch_cls_do01):
    do = 0.25


class base_fake(Base):
    use_fake_pressure = True


class base_better(Base):
    wandb_group = "betterConfig"
    use_bn_after_lstm = True
    use_RC_together = True
    loss_fnc = "huber"
    delta = 0.1
    fc = 64

class base_better_OP01(base_better):
    wandb_group = "betterConfig_OP01"
    out_phase_weight = 0.1

class base_better_OP01_noRCTogether(base_better_OP01):
    use_RC_together = False

class base_better_OP01_lossMAE(base_better_OP01):
    loss_fnc = "mae"

class base_better_OP01_WarmUp(base_better_OP01):
    scheduler = "cosineWithWarmUp"
    warmup = 20
    epochs = 200

class base_better2(base_better_OP01_lossMAE):
    wandb_group = "betterConfig_FE_dropCol"
    drop_useless_cols = True



class PulpFiction(base_better):
    model_module = "PulpFiction"
    hidden = [768, 512, 384, 256, 128]
    hidden_gru = [384, 256, 128, 64]
    fc = 128
    factor = 0.85
    patience = 7
    es = 21


class base_transformer(base_better2):
    model_module = "transformer"
    fc = 64
    hidden = [256, 128]



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
