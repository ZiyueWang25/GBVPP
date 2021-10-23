import time
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim.swa_utils import update_bn
import wandb

from util import *
from dataset import *
from models import get_model
from loss import get_loss
from metric import cal_mae_metric
from optim import get_optimizer


def training_loop(train_df, config):
    # train_df should already have features added
    if config.use_wandb:
        wandb.login(key=get_key(config.wandb_key_path))
        wandb.init(project=config.wandb_project, name=config.model_version + config.wandb_post,
                   config=class2dict(config), group=config.wandb_group, job_type=config.model_version)

    folds_val_score = []
    for fold in range(5):
        seed_torch(config.seed)
        print('Fold: ', fold)
        if fold not in config.train_folds:
            print("skip")
            continue
        best_valid_score = run_fold(fold, train_df.copy(), config)
        folds_val_score.append(best_valid_score)
    print('folds score:', folds_val_score)
    print("Avg: {:.5f}".format(np.mean(folds_val_score)))
    print("Std: {:.5f}".format(np.std(folds_val_score)))
    if config.use_wandb:
        wandb.finish()


def run_fold(fold, original_train_df, config, swa_start_step=None, swa_start_epoch=None, **kwargs):
    train_df = generate_PL(fold, original_train_df.copy(), config)
    X_train, y_train, w_train, X_valid, y_valid, w_valid = prepare_train_valid(train_df, config, fold)

    print('training data samples, val data samples: ', X_train.shape, X_valid.shape)
    w_train_transform = transform_weight(w_train, config)
    w_valid_transform = transform_weight(w_train, config)
    train_dt = VPP(X_train, y_train, w_train_transform, config)
    valid_dt = VPP(X_valid, y_valid, w_valid_transform, config)

    train_loader = DataLoader(train_dt,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dt,
                              batch_size=config.batch_size,
                              shuffle=False,
                              num_workers=config.num_workers, pin_memory=True, drop_last=False)

    model = get_model(X_train.shape[-1], config)
    print("Model Size: {}".format(get_n_params(model)))
    model.to(config.device)
    if len(config.gpu) > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, len(X_train), config)
    swa_model, swa_scheduler = None, None
    best_valid_score = np.inf
    if config.ckpt_folder is not None:
        print(f"Load Checkpoint from folder: {config.ckpt_folder}")
        checkpoint = torch.load(f'{config.ckpt_folder}/Fold_{fold}_best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        best_valid_score = float(checkpoint['best_valid_score'])
    criterion = get_loss(config)
    trainer = Trainer(model, optimizer, criterion, scheduler,
                      y_valid, w_valid,
                      best_valid_score, fold, config,
                      swa_model=swa_model, swa_scheduler=swa_scheduler,
                      swa_start_step=swa_start_step, swa_start_epoch=swa_start_epoch)

    trainer.fit(
        epochs=config.epochs,
        train_loader=train_loader,
        valid_loader=valid_loader,
        save_path=f'{config.model_output_folder}/Fold_{fold}_',
    )
    return trainer.best_valid_score


class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler,
                 y_valid, w_valid,
                 best_valid_score, fold, config,
                 swa_model=None, swa_scheduler=None, swa_start_step=None,
                 swa_start_epoch=None, **kwargs):
        self.model = model
        self.device = config.device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.best_valid_score = best_valid_score
        self.y_valid = y_valid
        self.w_valid = w_valid
        self.fold = fold
        self.max_grad_norm = 100

        self.do_reg = config.do_reg
        self.pressure_unique_path = config.pressure_unique_path

        # swa
        self.swa_model = swa_model
        self.swa_scheduler = swa_scheduler
        self.swa_start_epoch = swa_start_epoch
        self.swa_start_step = swa_start_step
        self.step = 0  # for swa

        # speed
        self.use_auto_cast = config.use_auto_cast

        # log
        self.print_num_steps = config.print_num_steps
        self.use_wandb = config.use_wandb
        self.es = config.es

    def fit(self, epochs, train_loader, valid_loader, save_path):
        train_losses = []
        valid_losses = []
        es_cnt = 0
        for n_epoch in range(epochs):
            start_time = time.time()
            print('Epoch: ', n_epoch)
            train_loss, lr = self.train_epoch(train_loader, n_epoch)
            valid_loss, valid_preds = self.valid_epoch(valid_loader, self.model)

            if self.swa_model is not None:
                if n_epoch >= self.swa_start_epoch:
                    print(f"Epoch {n_epoch}, update swa model")
                    self.swa_model.update_parameters(self.model)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(valid_loss)
            valid_score = cal_mae_metric(self.y_valid, valid_preds, self.w_valid)

            if self.best_valid_score > valid_score:
                self.best_valid_score = valid_score
                self.save_model(n_epoch, save_path + f'best_model.pth', valid_preds)
                es_cnt = 0
            else:
                es_cnt += 1

            print('loss:  {:.4f}, val_loss {:.4f}, val_score {:.4f}, best_val_score {:.4f}, lr {:.5f} '
                  '--- use {:.3f}s'.format(train_loss, valid_loss, valid_score, self.best_valid_score, lr,
                                           time.time() - start_time))
            if self.use_wandb:
                wandb.log({f"[fold{self.fold}] epoch": n_epoch + 1,
                           f"[fold{self.fold}] avg_train_loss": train_loss,
                           f"[fold{self.fold}] avg_val_loss": valid_loss,
                           f"[fold{self.fold}] val_score": valid_score,
                           f"[fold{self.fold}] best_val_score": self.best_valid_score,
                           })
                # save swa
            if es_cnt >= self.es:
                print("Early Stop")
                break

        if self.swa_model is not None:
            update_bn(train_loader, self.swa_model, device=self.device)
            valid_loss_swa, valid_preds_swa = self.valid_epoch(valid_loader, self.swa_model)
            valid_score_swa = cal_mae_metric(self.y_valid, valid_preds_swa, self.w_valid)
            print("SWA: Valid Loss {:.5f}, Valid Score {:.5f}".format(valid_loss_swa, valid_score_swa))
            if self.use_wandb:
                wandb.log({f"[fold{self.fold}] avg_val_loss_swa": valid_loss_swa,
                           f"[fold{self.fold}] val_score_swa": valid_score_swa})
                # update batch normalization
            save_dict = {
                "swa_model_state_dict": self.swa_model.state_dict(),
                "swa_scheduler": self.swa_scheduler.state_dict(),
                "valid_loss_swa": valid_loss_swa,
                "valid_score_swa": valid_score_swa,
            }
            torch.save(save_dict, save_path + f'swa_model.pth')

    def train_epoch(self, train_loader, n_epoch):
        scaler = GradScaler()
        self.model.train()
        losses = []
        train_loss = 0
        for step, batch in enumerate(train_loader, 1):
            self.step += 1
            self.optimizer.zero_grad()
            X = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            weights = batch[2].to(self.device)

            with autocast(enabled=self.use_auto_cast):
                outputs = self.model(X).squeeze()
                # get rid of NaN
                if self.do_reg:
                    loss = self.criterion(targets[outputs == outputs], outputs[outputs == outputs],
                                          weights[outputs == outputs])
                else:
                    loss = self.criterion(targets, outputs, weights)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()
            if not isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step()
            lr2 = self.optimizer.param_groups[0]['lr']
            loss2 = loss.detach()

            if self.use_wandb:
                wandb.log({f"[fold{self.fold}] loss": loss2,
                           f"[fold{self.fold}] lr": lr2,
                           "epoch": n_epoch,
                           "batch": step}
                          )

            losses.append(loss2)
            train_loss += loss2

        return train_loss.cpu().detach().numpy() / step, lr2

    def valid_epoch(self, valid_loader, model):
        model.eval()
        valid_loss = []
        preds = []
        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                weights = batch[2].to(self.device)
                outputs = model(X).squeeze()
                loss = self.criterion(targets, outputs, weights)
                valid_loss.append(loss.detach().item())
                preds.append(outputs.to('cpu').detach().numpy())
        predictions = np.concatenate(preds, axis=0)
        if not self.do_reg:
            predictions = cls_2_num_func(predictions.argmax(axis=-1), self.pressure_unique_path)
        return np.mean(valid_loss), predictions

    def save_model(self, n_epoch, save_path, valid_preds):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
                'scheduler': self.scheduler.state_dict(),
                'valid_preds': valid_preds,
            },
            save_path,
        )
