import os
import util
from argparse import ArgumentParser
from datetime import datetime

## TODO:
## 1. why mix-precision decrease performance? * 
## 2. use GRU, LSTM, transformer together *
## 3. add SWA
## 4. remove huge error cases (?)
## 5. KNN features
## 6. change the loss function to give 0.1 weight to samples * 
## 7. deal with NaN issue * 


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
    model_module = "BASE" # "RES"
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
    use_physics_fe = False
    drop_useless_cols = True

    # Model - rnn
    rnn_model = "LSTM"  # GRU
    hidden = [256] * 5
    rnn_do = 0
    # if we use residual block format, number of gru should be 1 smaller than lstm
    hidden_gru = [256] * 4
    
    # head 
    fc = 50
    use_ch = False
    ch_do = 0.1

    # Model - transformer
    use_transformer = False
    d_model = 256
    n_head = 8
    tsf_do = 0.2
    dim_forward = 2048
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
    out_phase_weight = 0.1
    loss_fnc = "huber" #huber
    delta = 0.25

    # swa
    use_swa = False
    swa_start_step = None
    swa_start_epoch = None
    swa_val_score_th = 0.2

    # logging
    use_wandb = True
    wandb_project = "GBVPP_newCV"
    wandb_post = ""
    wandb_group = None
    print_num_steps = 100

    # speed
    num_workers = 8
    use_auto_cast = True


class newStart(Base):
    wandb_group = "newStart"

class LSTM5_REG(newStart):
    pass

class LSTM5_REG_PL(LSTM5_REG):
    PL_folder = "/home/vincent/Kaggle/GBVPP/output/LSTM5_OP01_huber025_PL3/"

class LSTM7_REG_PL(LSTM5_REG):
    hidden = [256] * 7
    PL_folder = "/home/vincent/Kaggle/GBVPP/output/LSTM5_OP01_huber025_PL3/"


class LSTM5_CLS_DO02(newStart):
    do_reg = False
    loss_fnc = "ce"
    rnn_do = 0.2


class LSTM5_CLS_DO02_OP01(LSTM5_CLS_DO02):
    loss_fnc = "ce_custom"

class LSTM5_CLS_DO02_OP01_physics(LSTM5_CLS_DO02_OP01):
    use_physics_fe = True


class LSTM5_CLS_DO02_IPOnly(LSTM5_CLS_DO02_OP01):
    loss_fnc = "ce_custom"
    use_in_phase_only = True

class LSTM5_CLS_DO02_OP01_PL(LSTM5_CLS_DO02_OP01):
    PL_folder = "/home/vincent/Kaggle/GBVPP/output/LSTM5_OP01_huber025_PL3/"


class LSTM3_TSF2(newStart):
    use_transformer = True
    hidden = [256] * 3
    num_layer = 2

class LSTM3_TSF4_DO0(LSTM3_TSF2):
    tsf_do = 0
    num_layer = 4

class LSTM3_TSF4_DO0_dim512(LSTM3_TSF2):
    tsf_do = 0
    num_layer = 4
    d_model = 512




class LSTM3_TSFxx():
    PL_folder = "/home/vincent/Kaggle/GBVPP/output/LSTM5_OP01_huber025_PL3/"



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
