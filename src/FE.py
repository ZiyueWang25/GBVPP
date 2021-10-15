import pandas as pd


def add_features(df):
    ## TODO: Add features from public notebook

    df["step"] = list(range(80)) * (df.shape[0] // 80)
    # need robust process
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()

    # time
    df["time_step_diff"] = df.groupby('breath_id')['time_step'].diff().fillna(0.03343)
    df['cross_time'] = df['time_step'] * (1 - df['u_out'])

    # u_in
    g = df.groupby('breath_id')['u_in']
    df['cross_u_in'] = df['u_in'] * (1 - df['u_out'])
    # u_in diff
    df['u_in_diff'] = g.diff().fillna(0)
    df['u_in_diff_2'] = g.diff(2).fillna(0)
    df['u_in_diff_5'] = g.diff(5).fillna(0)
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

    # R C
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['R_C'] = df["R"] + '_' + df["C"]
    df = pd.get_dummies(df)


    df = df.fillna(0)
    # df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    # df['breath_id__u_in__mean'] = df.groupby(['breath_id'])['u_in'].transform('mean')
    # df['breath_id__u_out__mean'] = df.groupby(['breath_id'])['u_out'].transform('mean')
    #
    # df['breath_id__u_in__diffmax'] = df['breath_id__u_in__max'] - df['u_in']
    # df['breath_id__u_in__diffmean'] = df['breath_id__u_in__mean'] - df['u_in']
    return df