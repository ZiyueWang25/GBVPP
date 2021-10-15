import pandas as pd
import pickle
import torch
import numpy as np
from sklearn.preprocessing import RobustScaler

from FE import add_features
from pickle import dump
from pickle import load


class VPP(torch.utils.data.Dataset):
    def __init__(self, X, y, w):
        if y is None:
            y = np.zeros(len(X), dtype=np.float32)

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.w = w.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.w[i]


def changeType(data, is_train=True):
    dtypes = {"id": np.uint32, "breath_id": np.uint32, "R": np.uint32, "C": np.uint32,
              "time_step": np.float32, "u_in": np.float32, "u_out": np.int8}
    if is_train:
        dtypes["pressure"] = np.float32
    return data.astype(dtypes)


def read_data(config):
    n = 100*1024 if config.debug else None
    train = pd.read_csv(config.kaggle_data_folder + "/train.csv", nrows=n)
    test = pd.read_csv(config.kaggle_data_folder + "/test.csv", nrows=n)
    train = changeType(train, True)
    test = changeType(test, False)
    with open(config.input_folder + '/id_fold_dict.pickle', 'rb') as handle:
        id_fold_dict = pickle.load(handle)
    train['fold'] = train['id'].apply(lambda x: id_fold_dict[x])
    return train, test


def prepare_train_valid(train, valid, config, fold):
    print("Prepare train valid")
    print(train.shape, valid.shape)
    train = add_features(train)
    valid = add_features(valid)
    feature_cols = [col for col in train.columns if col not in ["id", "breath_id", "fold", "pressure"]]
    rs = RobustScaler(quantile_range=(config.low_q, config.high_q), unit_variance=config.unit_var)
    X_train = rs.fit_transform(train[feature_cols])

    # save scaler
    dump(rs, open(config.model_output_folder + f'/scaler_{fold}.pkl', 'wb'))

    X_valid = rs.transform(valid[feature_cols])
    print(X_train.shape)
    X_train = X_train.reshape(-1, 80, len(feature_cols))
    y_train = train['pressure'].values.reshape(-1, 80)
    w_train = 1 - train['u_out'].values.reshape(-1, 80)
    X_valid = X_valid.reshape(-1, 80, len(feature_cols))
    y_valid = valid['pressure'].values.reshape(-1, 80)
    w_valid = 1 - valid['u_out'].values.reshape(-1, 80)
    return X_train, y_train, w_train, X_valid, y_valid, w_valid


def prepare_test(test, config, fold):
    test = add_features(test)
    feature_cols = [col for col in test.columns if col not in ["id", "breath_id", "fold", "pressure"]]
    rs = load(open(config.model_output_folder + f'/scaler_{fold}.pkl', 'rb'))
    X_test = rs.transform(test[feature_cols])
    X_test = X_test.reshape(-1, 80, len(feature_cols))
    y_test = test['pressure'].values.reshape(-1, 80)
    w_test = 1 - test['u_out'].values.reshape(-1, 80)
    return X_test, y_test, w_test


def generate_PL(fold, train_df, config):
    if config.PL_folder is None:
        return train_df
    n = 100 * 1024 if config.debug else None
    PL_df = pd.read_csv(config.PL_folder + f"/test_Fold_{fold}.csv", nrows=n)
    PL_df = changeType(PL_df, False)
    PL_df["pressure"] = PL_df[f'preds_Fold_{fold}']
    PL_df['fold'] = -1
    PL_df = add_features(PL_df)
    PL_df = PL_df[train_df.columns]
    PL_df = pd.concat([train_df, PL_df]).reset_index(drop=True)
    PL_df.reset_index(inplace=True, drop=True)
    return PL_df
