from config import  Base

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

class Fork2(Base):
    # https://www.kaggle.com/tenffe/finetune-of-tensorflow-bidirectional-lstm
    wandb_group = "Fork2"
    optimizer = "Adam"
    batch_size = 1024
    low_q = 25
    high_q = 75
    unit_var = False
    strict_scale = False
    epochs = 300
    hidden = [1024, 512, 256, 128]
    fc = 128
    loss_fnc = "mae"
    es = 60
