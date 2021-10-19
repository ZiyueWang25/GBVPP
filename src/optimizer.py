from torch.optim import AdamW, Adam


def get_optimizer(model, config):
    if config.optimizer == "AdamW":
        optimizer = AdamW(model.parameters(), lr=config.lr, eps=1e-08, weight_decay=config.weight_decay, amsgrad=False)
    elif config.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=config.lr)
    return optimizer