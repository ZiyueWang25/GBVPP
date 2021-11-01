import pandas as pd
import numpy as np
from util import load_pickle
from tqdm import tqdm


def add_features_choice(df, config):
    if config.fe_type == "own":
        return add_features(df, config)
    elif config.fe_type == "fork":
        return add_features_fork(df, config)


def add_features(df, config):
    df["step"] = list(range(80)) * (df.shape[0] // 80)

    # u_out
    print("--- Generate u_out features ---")
    df['u_out_diff'] = df.groupby("breath_id")['u_out'].diff().fillna(0)
    for shift in range(1, 3):
        df[f'u_out_diff_back{shift}'] = df.groupby('breath_id')['u_out_diff'].shift(-shift)
    df.fillna(0, inplace=True)

    # time
    print("--- Generate time features ---")
    df['cross_time'] = df['time_step'] * (1 - df['u_out'])
    df['time_delta'] = df.groupby("breath_id")['time_step'].diff().fillna(0.033098770961621796)

    # u_in
    print("--- Generate u_in features ---")
    df.sort_values(["breath_id","id"], inplace=True)
    g = df.groupby('breath_id')['u_in']
    df['cross_u_in'] = df['u_in'] * (1 - df['u_out'])
    # u_in: first point, last point
    df['u_in_first'] = g.transform(lambda s: s.iloc[0])
    df['u_in_last'] = g.transform(lambda s: s.iloc[-1])
    df['time_end'] = df.groupby("breath_id")["time_step"].transform(lambda s: s.iloc[-1])
    # shift
    for shift in range(1, 5):
        df[f'u_in_lag{shift}'] = g.shift(shift)
        df[f'u_in_lag_back{shift}'] = g.shift(-shift)
        df[f'u_in_diff{shift}'] = df["u_in"] - df[f'u_in_lag{shift}']
        df[f'u_in_diff_back{shift}'] = df["u_in"] - df[f'u_in_lag_back{shift}']
    df.fillna(0, inplace=True)
    # cumsum
    df['u_in_cumsum'] = g.cumsum()
    df['tmp'] = df['time_delta'] * df['u_in']
    df['area'] = df.groupby('breath_id')['tmp'].cumsum()
    df.drop(columns=["tmp"], inplace=True)
    # expanding
    df['ewm_u_in_mean'] = g.ewm(halflife=10).mean().reset_index(level=0, drop=True)
    df['ewm_u_in_std'] = g.ewm(halflife=10).std().reset_index(level=0, drop=True)
    # rolling
    df['rolling_10_mean'] = g.rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    df['rolling_10_max'] = g.rolling(window=10, min_periods=1).max().reset_index(level=0, drop=True)
    df['rolling_10_std'] = g.rolling(window=10, min_periods=5).std().reset_index(level=0, drop=True).fillna(0)
    # expanding
    df['expand_mean'] = g.expanding(1).mean().reset_index(level=0, drop=True)
    df['expand_median'] = g.expanding(1).median().reset_index(level=0, drop=True)
    df['expand_std'] = g.expanding(5).std().reset_index(level=0, drop=True).fillna(0)
    df['expand_max'] = g.expanding(1).max().reset_index(level=0, drop=True)
    df['expand_skew'] = g.expanding(1).skew().reset_index(level=0, drop=True)
    df['expand_kurt'] = g.expanding(1).kurt().reset_index(level=0, drop=True)
    # transform
    df['u_in_max'] = g.transform('max')
    df['u_in_mean'] = g.transform('mean')
    df['u_in_diffmax'] = df['u_in_max'] - df['u_in']
    df['u_in_diffmean'] = df['u_in_mean'] - df['u_in']

    if config.use_crossSectional_features:
        df.sort_values(["breath_id","id"], inplace=True)
        # Cross Sectional
        print("--- Generate cross sectional features ---")
        RC_u_in_median = pd.read_csv(config.RC_u_in_median_path).set_index(["R", "C", "step"])
        RC_u_in_mean = pd.read_csv(config.RC_u_in_mean_path).set_index(["R", "C", "step"])
        df = df.merge(RC_u_in_median, left_on=["R", "C", "step"], right_index=True)
        df = df.merge(RC_u_in_mean, left_on=["R", "C", "step"], right_index=True)
        df["RC_u_in_median_diff"] = df["u_in"] - df["RC_u_in_median"]
        df["RC_u_in_mean_diff"] = df["u_in"] - df["RC_u_in_mean"]
        df["RC_u_in_median_diff_cum"] = df.groupby("breath_id")["RC_u_in_median_diff"].cumsum()
        df["RC_u_in_mean_diff_cum"] = df.groupby("breath_id")["RC_u_in_mean_diff"].cumsum()
        for lag in range(1, 3):
            df[f'RC_u_in_median_diff_shift{lag}'] = df.groupby("breath_id")['RC_u_in_median_diff'].shift(lag)
            df[f'RC_u_in_median_diff_shift_back{lag}'] = df.groupby("breath_id")['RC_u_in_median_diff'].shift(-lag)

    # fake pressure
    if config.use_fake_pressure:
        df.sort_values(["breath_id","id"], inplace=True)
        print("--- Generate fake pressure features ---")
        g = df.groupby("breath_id")["pressure"]
        for i in range(1, 4):
            df[f"pressure_prev_lag{i}"] = g.shift(i)
            df[f"pressure_prev_lag_back{i}"] = g.shift(-i)
        for i in range(1, 3):
            df[f"pressure_prev_lag{i}_minus_lag{i+1}"] = df[f"pressure_prev_lag{i}"] - df[f"pressure_prev_lag{i+1}"]
            df[f"pressure_prev_lag_back{i+1}_minus_lag_back{i}"] = df[f"pressure_prev_lag_back{i+1}"] - df[f"pressure_prev_lag_back{i}"]
    # rate
    df['u_in_rate'] = df['u_in_diff1'] / df['time_delta']
    
    # physics feature
    if config.use_physics_fe:
        print("--- Generate physics features ---")
        df.sort_values(["breath_id","id"], inplace=True)
        df['area_2'] = df['time_step'] * df['u_in']
        df['area_2'] = df.groupby('breath_id')['area_2'].cumsum()
        df['time_step_cumsum'] = df.groupby(['breath_id'])['time_step'].cumsum()
        df['exponent']=(- df['time_step'])/(df['R']*df['C'])
        df['factor']= np.exp(df['exponent'])
        df['vf']=(df['u_in_cumsum']*df['R'])/df['factor']
        df['vt']=0
        df.loc[df['time_step'] != 0, 'vt']=df['area']/(df['C']*(1 - df['factor']))
        df['v']=df['vf']+df['vt']        
        df.drop(columns=["exponent", "factor"], inplace=True)
        
    if config.use_cluster:
        print("--- Generate Cluster features ---")
        df.sort_values(["breath_id","id"], inplace=True)
        for n_cluster in tqdm([10, 15, 25, 40]):
            u_in_median = load_pickle(config.input_folder + "/" + f"cluster_u_in_median_{n_cluster}.pkl")
            u_in_mean = load_pickle(config.input_folder + "/" + f"cluster_u_in_median_{n_cluster}.pkl")
            id_cluster = load_pickle(config.input_folder + "/" + f"cluster_id_{n_cluster}.pkl")
            df[f"cluster_{n_cluster}_median_diff"] = df.groupby("breath_id", group_keys=False).apply(lambda df: df.u_in - u_in_median[id_cluster[df.breath_id.iloc[0]]])
            df[f"cluster_{n_cluster}_mean_diff"] = df.groupby("breath_id", group_keys=False).apply(lambda df: df.u_in - u_in_mean[id_cluster[df.breath_id.iloc[0]]])

    # R C
    print("--- Generate R C features ---")
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    if config.use_RC_together:
        df['R_C'] = df["R"] + '_' + df["C"]
    df = pd.get_dummies(df)
    # rearange columns
    RC_cols = ['R_20', 'R_5', 'R_50', 'C_10', 'C_20', 'C_50',
               'R_C_20_10', 'R_C_20_20', 'R_C_20_50', 'R_C_50_10', 'R_C_50_20',
               'R_C_50_50', 'R_C_5_10', 'R_C_5_20', 'R_C_5_50']
    for col in RC_cols:
        if col not in df.columns:
            df[col] = 0
    None_RC_cols = [col for col in df.columns if col not in RC_cols]
    df = df[None_RC_cols + RC_cols]

    if config.drop_useless_cols:
        drop_cols = ["step", "cross_time", "expand_skew", "expand_kurt"]
        print("Drop Low Importance Columns:", drop_cols)
        df.drop(columns=drop_cols, inplace=True)

    df = df.fillna(0)
    df.sort_values(["breath_id","id"], inplace=True)
    return df


def add_features_fork(df,config):
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df['u_in_lag2'] = df['u_in'].shift(2).fillna(0)
    df['u_in_lag4'] = df['u_in'].shift(4).fillna(0)
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df = pd.get_dummies(df)

    g = df.groupby('breath_id')['u_in']
    df['ewm_u_in_mean'] = g.ewm(halflife=10).mean()\
                           .reset_index(level=0, drop=True)
    df['ewm_u_in_std'] = g.ewm(halflife=10).std()\
                          .reset_index(level=0, drop=True)
    df['ewm_u_in_corr'] = g.ewm(halflife=10).corr()\
                           .reset_index(level=0, drop=True)

    df['rolling_10_mean'] = g.rolling(window=10, min_periods=1).mean()\
                             .reset_index(level=0, drop=True)
    df['rolling_10_max'] = g.rolling(window=10, min_periods=1).max()\
                            .reset_index(level=0, drop=True)
    df['rolling_10_std'] = g.rolling(window=10, min_periods=1).std()\
                            .reset_index(level=0, drop=True)

    df['expand_mean'] = g.expanding(2).mean()\
                         .reset_index(level=0, drop=True)
    df['expand_max'] = g.expanding(2).max()\
                        .reset_index(level=0, drop=True)
    df['expand_std'] = g.expanding(2).std()\
                        .reset_index(level=0, drop=True)
    df = df.fillna(0)

    return df