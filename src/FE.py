import pandas as pd

def add_features_choice(df, config):
    if config.fe_type == "own":
        return add_features(df)
    elif config.fe_type == "fork":
        return add_features_fork(df)

def add_features(df):
    ## TODO: Add features from public notebook
    df["step"] = list(range(80)) * (df.shape[0] // 80)
    # need robust process
    df['time_delta'] = df['time_step'].diff().fillna(0)
    df['time_delta'].mask(df['time_delta'] < 0, 0, inplace=True)
    df['tmp'] = df['time_delta'] * df['u_in']
    df['area'] = df.groupby('breath_id')['tmp'].cumsum()
    df.drop(columns=["tmp"], inplace=True)
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()

    # u_out
    df['u_out_diff'] = df.groupby("breath_id")['u_out'].diff().fillna(0)

    # time
    df['cross_time'] = df['time_step'] * (1 - df['u_out'])

    # u_in
    g = df.groupby('breath_id')['u_in']
    df['cross_u_in'] = df['u_in'] * (1 - df['u_out'])
    # u_in: first point, last point
    first_df = df.iloc[0::80, :]
    last_df = df.iloc[79::80, :]
    u_in_first_dict = dict(zip(first_df['breath_id'], first_df['u_in']))
    df['u_in_first'] = df['breath_id'].map(u_in_first_dict)
    u_in_last_dict = dict(zip(first_df['breath_id'], last_df['u_in']))
    df['u_in_last'] = df['breath_id'].map(u_in_last_dict)
    del u_in_first_dict, u_in_last_dict
    time_end_dict = dict(zip(last_df['breath_id'], last_df['time_step']))
    df['time_end'] = df['breath_id'].map(time_end_dict)
    del last_df, time_end_dict
    # shift
    for shift in range(1,5):
        df[f'u_in_lag{shift}'] = g.shift(1)
        df[f'u_in_lag_back{shift}'] = g.shift(-shift)
    df.fillna(0,inplace=True)
    # u_in diff
    df['u_in_diff'] = g.diff().fillna(0)
    df['u_in_diff_2'] = g.diff(2).fillna(0)
    df['u_in_diff_4'] = g.diff(4).fillna(0)
    # expanding
    df['ewm_u_in_mean'] = g.ewm(halflife=10).mean().reset_index(level=0, drop=True)
    df['ewm_u_in_std'] = g.ewm(halflife=10).std().reset_index(level=0, drop=True)
    df['ewm_u_in_corr'] = g.ewm(halflife=10).corr().reset_index(level=0, drop=True)
    # rolling
    df['rolling_10_mean'] = g.rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    df['rolling_10_max'] = g.rolling(window=10, min_periods=1).max().reset_index(level=0, drop=True)
    df['rolling_10_std'] = g.rolling(window=10, min_periods=1).std().reset_index(level=0, drop=True)
    # expanding
    df['expand_mean'] = g.expanding(1).mean().reset_index(level=0, drop=True)
    df['expand_median'] = g.expanding(1).median().reset_index(level=0, drop=True)
    df['expand_std'] = g.expanding(1).std().reset_index(level=0, drop=True)
    ## TODO: add expand_mad
    df['expand_max'] = g.expanding(1).max().reset_index(level=0, drop=True)
    df['expand_skew'] = g.expanding(1).kurt().reset_index(level=0, drop=True)
    df['expand_kurt'] = g.expanding(1).kurt().reset_index(level=0, drop=True)

    # Cross Sectional
    RC_u_in_median = df.groupby(["R", "C", "step"])["u_in"].median()
    RC_u_in_mean = df.groupby(["R", "C", "step"])["u_in"].mean()
    df = df.merge(RC_u_in_median.to_frame("RC_u_in_median"), left_on=["R", "C", "step"], right_index=True)
    df = df.merge(RC_u_in_mean.to_frame("RC_u_in_mean"), left_on=["R", "C", "step"], right_index=True)
    df["RC_u_in_median_diff"] = df["u_in"] - df["RC_u_in_median"]
    df["RC_u_in_mean_diff"] = df["u_in"] - df["RC_u_in_mean"]
    df.drop(columns=["RC_u_in_median", "RC_u_in_mean"], inplace=True)

    df.sort_values("id", inplace=True)

    # R C
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df = pd.get_dummies(df)


    df = df.fillna(0)
    # df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    # df['breath_id__u_in__mean'] = df.groupby(['breath_id'])['u_in'].transform('mean')
    # df['breath_id__u_out__mean'] = df.groupby(['breath_id'])['u_out'].transform('mean')
    #
    # df['breath_id__u_in__diffmax'] = df['breath_id__u_in__max'] - df['u_in']
    # df['breath_id__u_in__diffmean'] = df['breath_id__u_in__mean'] - df['u_in']
    return df


def add_features_fork(df):
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