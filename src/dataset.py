import pandas as pd
import pickle
import torch
import numpy as np
from sklearn.preprocessing import RobustScaler

from FE import add_features

from pickle import dump, load


class VPP(torch.utils.data.Dataset):
    def __init__(self, X, y, w, config=None):

        self.X = X.astype(np.float32)
        self.w = w.astype(np.float32)

        if y is None:
            self.y = np.zeros(len(X), dtype=np.float32)
        elif config.do_reg:
            self.y = y.astype(np.float32)
        elif not config.do_reg:
            self.y = num_2_cls_func(y, config.pressure_unique_path)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.w[i]


class LR_VPP(torch.utils.data.Dataset):
    def __init__(self, X, y, w, config=None):
        if y is None:
            y = np.zeros(len(X), dtype=np.float32)

        self.X = X.astype(np.float32)
        self.w = w.astype(np.float32)
        if config.do_reg:
            self.y = y.astype(np.float32)
        elif not config.do_reg:
            self.y = num_2_cls_func(y, config.pressure_unique_path)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.w[i]


def read_data(config):
    n = 100*1024 if config.debug else None
    dict_types = {
        'id': np.uint32,
        'breath_id': np.uint32,
        'R': np.uint32,
        'C': np.uint32,
        'time_step': np.float32,
        'u_in': np.float32,
        'u_out': np.int8,
    }
    train = pd.read_csv(config.kaggle_data_folder + "/train.csv", nrows=n, dtype=dict_types)
    test = pd.read_csv(config.kaggle_data_folder + "/test.csv", nrows=n, dtype=dict_types)
    with open(config.input_folder + '/id_fold_dict.pickle', 'rb') as handle:
        id_fold_dict = pickle.load(handle)
    train['fold'] = train['breath_id'].apply(lambda x: id_fold_dict[x])
    return train, test


def prepare_train_valid(train_df, config, fold):
    train_index, valid_index = train_df.query(f"fold!={fold}").index, train_df.query(f"fold=={fold}").index
    rs = RobustScaler(quantile_range=(config.low_q, config.high_q), unit_variance=config.unit_var)
    train, valid = train_df.iloc[train_index], train_df.iloc[valid_index]
    if config.strict_scale:
        print("Use scale to fit train and scale valid")
        print("Prepare train valid")
        print(train.shape, valid.shape)
        feature_cols = [col for col in train.columns if col not in ["id", "breath_id", "fold", "pressure"]]

        X_train = rs.fit_transform(train[feature_cols])
        X_valid = rs.transform(valid[feature_cols])

        dump(rs, open(config.model_output_folder + f'/scaler_{fold}.pkl', 'wb'))
    else:
        print("Unsctrict scale - leakage..")
        feature_cols = [col for col in train_df.columns if col not in ["id", "breath_id", "fold", "pressure"]]
        X_all = rs.fit_transform(train_df[feature_cols])
        X_train, X_valid = X_all[train_index, :], X_all[valid_index, :]
        dump(rs, open(config.model_output_folder + f'/scaler.pkl', 'wb'))

    X_train = X_train.reshape(-1, 80, len(feature_cols))
    y_train = train['pressure'].values.reshape(-1, 80)
    w_train = 1 - train['u_out'].values.reshape(-1, 80)
    X_valid = X_valid.reshape(-1, 80, len(feature_cols))
    y_valid = valid['pressure'].values.reshape(-1, 80)
    w_valid = 1 - valid['u_out'].values.reshape(-1, 80)
    return X_train, y_train, w_train, X_valid, y_valid, w_valid



def prepare_test(test, config, fold):
    # test data should already have features
    feature_cols = [col for col in test.columns if col not in ["id", "breath_id", "fold", "pressure"]]
    if config.strict_scale:
        rs = load(open(config.model_output_folder + f'/scaler_{fold}.pkl', 'rb'))
    else:
        rs = load(open(config.model_output_folder + f'/scaler.pkl', 'rb'))

    X_test = rs.transform(test[feature_cols])
    X_test = X_test.reshape(-1, 80, len(feature_cols))
    y_test = test['pressure'].values.reshape(-1, 80)
    w_test = 1 - test['u_out'].values.reshape(-1, 80)
    return X_test, y_test, w_test


def generate_PL(fold, train_df, config):
    if config.PL_folder is None:
        return train_df
    print("----- USE PL ---- ")
    pressure_unique = np.load(config.pressure_unique_path)
    n = 100 * 1024 if config.debug else None
    dict_types = {
        'id': np.uint32,
        'breath_id': np.uint32,
        'R': np.uint32,
        'C': np.uint32,
        'time_step': np.float32,
        'u_in': np.float32,
        'u_out': np.int8,
    }
    PL_df = pd.read_csv(config.kaggle_data_folder + "/test.csv", nrows=n, dtype=dict_types)
    PL_df["pressure"] = pd.read_csv(config.PL_folder + f"/test_fold{fold}.csv", nrows=n)[f'preds_fold{fold}']
    if len(PL_df.pressure.unique()) > 950:
        print("Rounding....")
        PL_df["pressure"] = PL_df["pressure"].map(lambda x: pressure_unique[np.abs(pressure_unique-x).argmin()])
    PL_df['fold'] = -1
    PL_df = add_features(PL_df)
    PL_df = PL_df[train_df.columns]
    PL_df = pd.concat([train_df, PL_df]).reset_index(drop=True)
    PL_df.reset_index(inplace=True, drop=True)
    return PL_df


def num_2_cls_func(y, pressure_unique_path):
    print("Transform number into class")
    pressure_unique = np.load(pressure_unique_path)
    num_to_cls_dict = dict(zip(pressure_unique, list(range(len(pressure_unique)))))
    num_to_cls_func = np.vectorize(lambda x: num_to_cls_dict[x] if x != -999 else x)
    return num_to_cls_func(y)


def cls_2_num_func(y, pressure_unique_path):
    print("Transform class into number")
    pressure_unique = np.load(pressure_unique_path)
    cls_to_num_dict = dict(zip(list(range(len(pressure_unique))), pressure_unique))
    cls_to_num_func = np.vectorize(lambda x: cls_to_num_dict[x])
    return cls_to_num_func(y)

