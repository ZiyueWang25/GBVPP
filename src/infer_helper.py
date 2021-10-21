from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel
from dataset import *
from models import get_model


def get_pred(loader, model, device):
    preds = []
    for step, batch in enumerate(loader, 1):
        if step % 500 == 0:
            print("step {}/{}".format(step, len(loader)))
        with torch.no_grad():
            X = batch[0].to(device)
            outputs = model(X)
            outputs = outputs.squeeze().cpu().detach().numpy()
            preds.append(outputs)
    predictions = np.concatenate(preds)
    predictions = predictions.flatten()
    return predictions


def get_cv_score(config):
    cv_scores = []
    for fold in tqdm(config.train_folds):
        checkpoint = torch.load(f'{config.model_output_folder}/Fold_{fold}_best_model.pth')
        cv_scores.append(checkpoint['best_valid_score'])
    print("CV_Scores:", cv_scores)
    print("Mean : {:.5f}".format(np.mean(cv_scores)))
    print("Std : {:.5f}".format(np.std(cv_scores)))
    return np.mean(cv_scores)


def removeDPModule(state_dict):
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def get_test_avg(test_df, config, cv):
    test_df['pressure'] = 0
    test_avg = test_df[['id', 'pressure']].copy()
    cv_str = "{:.0f}".format(cv * 1e5)
    for fold in tqdm(config.train_folds):
        X_test, y_test, w_test = prepare_test(test_df, config, fold)
        data_retriever = VPP(X_test, y_test, w_test)
        data_loader = DataLoader(data_retriever,
                                 batch_size=config.batch_size//2,
                                 shuffle=False,
                                 num_workers=config.num_workers, pin_memory=True, drop_last=False)

        model = get_model(X_test.shape[-1], config)
        if config.use_swa:
            swa_model = AveragedModel(model)
            checkpoint = torch.load(f'{config.model_output_folder}/Fold_{fold}_swa_model.pth')
            model = swa_model
            model.load_state_dict(removeDPModule(checkpoint['model_swa_state_dict']))
        else:
            checkpoint = torch.load(f'{config.model_output_folder}/Fold_{fold}_best_model.pth')
            model.load_state_dict(removeDPModule(checkpoint['model_state_dict']))
        model.to(device=config.device)
        if len(config.gpu) > 1 and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.eval()
        test_avg[f"preds_fold{fold}"] = get_pred(data_loader, model, config.device)
        test_avg["pressure"] = test_avg["pressure"] + test_avg[f"preds_fold{fold}"] / len(config.train_folds)
        test_avg[["id", f"preds_fold{fold}"]].to_csv(config.model_output_folder + f"/test_fold{fold}.csv",
                                                     index=False)
    test_avg.to_csv(config.model_output_folder + f"/test_pred_all_{cv_str}.csv", index=False)
    test_avg[['id', 'pressure']].to_csv(config.model_output_folder + f"/submission_{cv_str}.csv", index=False)
    print(test_avg['pressure'].describe())
    print("test file saved to:", config.model_output_folder + f"/submission_{cv_str}.csv")
    return test_avg
