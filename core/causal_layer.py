from causalml.inference.meta import BaseXRegressor, BaseTRegressor, BaseSRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
import numpy as np
import pandas as pd
import random
import statsmodels.formula.api as smf
# from sklearn.ensemble import RandomForestClassifier as RFR
# learn = BaseSRegressor()
# learn.estimate_ate()


def x_learner(X, T, Y):
    if X is None:
        X = np.array([[1]]*len(Y))
    learner_x = BaseXRegressor(XGBRegressor())

    te, lb, ub = learner_x.estimate_ate(X=X, treatment=T, y=Y)
    return te[0], lb[0], ub[0]


def t_learner(X, treatment, y):
    if X is None:
        X = np.array([[1]]*len(y))
    learner_t = BaseTRegressor(RandomForestRegressor())
    te, lb, ub = learner_t.estimate_ate(X=X, treatment=treatment, y=y)
    return te[0], lb[0], ub[0]


def avg_cause_effect(df, treatment, Y):
    '''
    :param df:
    :param treatment: col name of treatment
    :param Y: col name of Y
    :return:
    '''
    assert treatment in df.columns and Y in df.columns
    if len(df.columns) == 2:
        X = None
    else:
        X = df.drop([treatment, Y], axis=1)
    return x_learner(X, df[treatment], df[Y])


def regression_analysis(df, X, Y, ref=None):
    '''
    :param df:
    :param X: one col name or list of col name
    :param Y: col name of Y
    :param ref: {col: value, col: value, ..., } control for discrete variable
    :return:
    '''
    def cause_effect_by_excute(excute_comment, df, logist_flag=True):
        result = smf.logit(excute_comment, data=df).fit() if logist_flag else smf.ols(excute_comment, data=df).fit()

        # ... Define and fit model
        odds_ratios = pd.DataFrame(
            {
                "OR": result.params,
                "Lower CI": result.conf_int()[0],
                "Upper CI": result.conf_int()[1],
            }
        )
        odds_ratios = np.exp(odds_ratios)
        odds_ratios["std err"] = result.bse
        odds_ratios["pvalue"] = result.pvalues
        # display(odds_ratios.round(4))
        return odds_ratios
    X = [X] if type(X) is not list else X
    excute_comment = Y + " ~ "
    # + " + ".join(X) if len(X) > 1 else Y + " ~ "

    for i, x in enumerate(X):
        if ref and x in ref.keys():
            excute_comment += (" + "if i else "")+"C(" + x + ", Treatment(reference='" + str(ref[x]) + "'))"
        else:
            excute_comment += (" + "if i else "")+x
    print("Regression Formula: ", excute_comment)
    # print(df.columns)
    # input()
    # max([len(df[col].unique()) for col in df.columns])

    return cause_effect_by_excute(excute_comment, df, len(df[Y].unique()) == 2)


def fit_on_predict_on(df1, df2, Y, treatment='T'):
    '''
    :param df1:
    :param X2: Dataframe or numpy.array
    :param Y: col name
    :param treatment: col name
    :return:
    '''

    if len([col for col in df1.columns if col not in [treatment, Y]]) == 0:
        X1 = np.array([[1]] * len(df1))
        X2 = np.array([[1]] * len(df2))
    else:
        X1 = df1.drop([treatment, Y], axis=1).to_numpy()
        X2 = df2.drop([treatment, Y], axis=1).to_numpy()

    Y1 = df1[Y].to_numpy()
    Y2 = df2[Y].to_numpy()

    T1 = df1[treatment].to_numpy()
    T2 = df2[treatment].to_numpy()

    learner_x = BaseXRegressor(XGBRegressor())
    learner_x.fit(X=X1, treatment=T1, y=Y1)

    learner_x._set_propensity_models(X=X2, treatment=T2, y=Y2)  # compute propensity score based df2
    te, dhat_cs, dhat_ts = learner_x.predict(X2, treatment=T2, y=Y2, p=learner_x.propensity, return_components=True)    # propensity is trained here

    _ate = te[:, 0].mean()
    dhat_c = dhat_cs[1]
    dhat_t = dhat_ts[1]

    p_filt = learner_x.propensity[1]
    w = (df2[treatment].to_numpy() == 1).astype(int)
    prob_treatment = float(sum(w)) / w.shape[0]
    se = np.sqrt(
        (
                learner_x.vars_t[1] / prob_treatment
                + learner_x.vars_c[1] / (1 - prob_treatment)
                + (p_filt * dhat_c + (1 - p_filt) * dhat_t).var()
        )
        / w.shape[0]
    )
    _ate_lb = _ate - se * norm.ppf(1 - 0.05 / 2)
    _ate_ub = _ate + se * norm.ppf(1 - 0.05 / 2)

    return _ate, _ate_lb, _ate_ub


def avg_cause_effect(df: pd.DataFrame, treatment, Y):
    assert len(df[treatment].unique()) == 2
    if len(df.columns) == 2:
        X = np.array([[1]] * len(df))
    else:
        X = df.drop([treatment, Y], axis=1).to_numpy()
    return x_learner(X, df[treatment].to_numpy(), df[Y].to_numpy())


def relational_cause_effect_xlearner(df: pd.DataFrame, treatment, Y):
    assert len(df[treatment+"_ego"].unique()) == 2 and len(df[treatment+"_peers"].unique()) == 2

    def construct_T(df1, df2):
        df1 = df1.drop([treatment+"_ego", treatment+"_peers"], axis=1)
        df1['T'] = [1 for _ in range(len(df1))]
        df2 = df2.drop([treatment+"_ego", treatment+"_peers"], axis=1)
        df2['T'] = [0 for _ in range(len(df2))]
        df = pd.concat([df1, df2])
        return df

    df11 = df.loc[(df[treatment+"_ego"] == 1) & (df[treatment+"_peers"] == 1)]
    df10 = df.loc[(df[treatment+"_ego"] == 1) & (df[treatment+"_peers"] == 0)]
    df01 = df.loc[(df[treatment+"_ego"] == 0) & (df[treatment+"_peers"] == 1)]
    df00 = df.loc[(df[treatment+"_ego"] == 0) & (df[treatment+"_peers"] == 0)]

    if not (len(df11) > 0 and len(df10) > 0 and len(df01) > 0 and len(df00) > 0):
        raise Exception("Cannot do causal inference, lack of data length: df11", len(df11), ", df10", len(df10), ", df01", len(df01), ", df00", len(df00))

    # TODO isolated treatment effect
    df1101 = construct_T(df11, df01)
    df1000 = construct_T(df10, df00)
    df_new = pd.concat([df1101, df1000])
    # 11 - 01
    ite1, ite_lb1, ite_ub1 = fit_on_predict_on(df1101, df_new, Y=Y, treatment='T')
    # 10-00
    ite2, ite_lb2, ite_ub2 = fit_on_predict_on(df1000, df_new, Y=Y, treatment='T')
    # (11 + 10) - (01 + 00)
    ite3, ite_lb3, ite_ub3 = fit_on_predict_on(df_new, df_new, Y=Y, treatment='T')

    # ite, ite_lb, ite_ub = np.mean([ite1, ite2]), np.mean([ite_lb1, ite_lb2]), np.mean([ite_ub1, ite_ub2])

    # TODO relational treatment effect
    df1110 = construct_T(df11, df10)
    df0100 = construct_T(df01, df00)
    df_new = pd.concat([df1110, df0100])
    # 11 - 10
    rte1, rte_lb1, rte_ub1 = fit_on_predict_on(df1110, df_new, Y=Y, treatment='T')
    # 01 - 00
    rte2, rte_lb2, rte_ub2 = fit_on_predict_on(df0100, df_new, Y=Y, treatment='T')
    # (11 + 10) - (01 + 00)
    rte3, rte_lb3, rte_ub3 = fit_on_predict_on(df_new, df_new, Y=Y, treatment='T')

    # rte, rte_lb, rte_ub = np.mean([rte1, rte2]), np.mean([rte_lb1, rte_lb2]), np.mean([rte_ub1, rte_ub2])

    # overal treament effect
    # 11 - 00
    df1100 = construct_T(df11, df00)
    df1001 = construct_T(df10, df01)
    df1001['T'] = [random.randint(0, 1) for _ in range(len(df1001))]    # TODO or 10 -> T, 01 -> Not T
    df_new = pd.concat([df1100, df1001])

    tte1 = fit_on_predict_on(df1100, df_new, Y=Y, treatment='T')

    return np.mean([ite1, ite2, ite3]), np.mean([rte1, rte2, rte3]), tte1
    # return np.mean(list(ie0) + list(ie1)), np.mean(list(re0) + list(re1)), np.mean(oe)


def relational_cause_effect_tlearner(df: pd.DataFrame, treatment, Y):
    assert len(df[treatment + "_ego"].unique()) == 2 and len(df[treatment + "_peers"].unique()) == 2

    if len([col for col in df.columns if treatment not in col and Y not in col]) == 0:
        raise Exception("there is no covariate for Tlearner, suggest using SLearner")

    rfs11 = XGBRegressor()  # RFR(n_estimators=50)
    rfs10 = XGBRegressor()  # RFR(n_estimators=50)
    rfs01 = XGBRegressor()  # RFR(n_estimators=50)
    rfs00 = XGBRegressor()  # RFR(n_estimators=50)

    df11 = df.loc[(df[treatment + "_ego"] == 1) & (df[treatment + "_peers"] == 1)]
    df10 = df.loc[(df[treatment + "_ego"] == 1) & (df[treatment + "_peers"] == 0)]
    df01 = df.loc[(df[treatment + "_ego"] == 0) & (df[treatment + "_peers"] == 1)]
    df00 = df.loc[(df[treatment + "_ego"] == 0) & (df[treatment + "_peers"] == 0)]

    if not (len(df11) > 0 and len(df10) > 0 and len(df01) > 0 and len(df00) > 0):
        raise Exception("Cannot do causal inference, lack of data length: df11", len(df11), ", df10", len(df10), ", df01", len(df01), ", df00", len(df00))

    rfs11 = rfs11.fit(df11.drop([treatment+"_ego", treatment+"_peers", Y], axis=1), df11[Y])
    rfs10 = rfs10.fit(df10.drop([treatment+"_ego", treatment+"_peers", Y], axis=1), df10[Y])
    rfs01 = rfs01.fit(df01.drop([treatment+"_ego", treatment+"_peers", Y], axis=1), df01[Y])
    rfs00 = rfs00.fit(df00.drop([treatment+"_ego", treatment+"_peers", Y], axis=1), df00[Y])

    df_new = df.drop([treatment+"_ego", treatment+"_peers", Y], axis=1)
    ie1 = rfs11.predict(df_new) - rfs01.predict(df_new)
    ie0 = rfs10.predict(df_new) - rfs00.predict(df_new)

    re1 = rfs11.predict(df_new) - rfs10.predict(df_new)
    re0 = rfs01.predict(df_new) - rfs00.predict(df_new)

    oe = rfs11.predict(df_new) - rfs00.predict(df_new)

    return np.mean(list(ie0) + list(ie1)), np.mean(list(re0) + list(re1)), np.mean(oe)


def relational_cause_effect_slearner(df: pd.DataFrame, treatment, Y):
    assert len(df[treatment + "_ego"].unique()) == 2 and len(df[treatment + "_peers"].unique()) == 2
    rfs11 = XGBRegressor()  # RFR(n_estimators=50)
    rfs10 = XGBRegressor()  # RFR(n_estimators=50)
    rfs01 = XGBRegressor()  # RFR(n_estimators=50)
    rfs00 = XGBRegressor()  # RFR(n_estimators=50)

    df11 = df.loc[(df[treatment + "_ego"] == 1) & (df[treatment + "_peers"] == 1)]
    df10 = df.loc[(df[treatment + "_ego"] == 1) & (df[treatment + "_peers"] == 0)]
    df01 = df.loc[(df[treatment + "_ego"] == 0) & (df[treatment + "_peers"] == 1)]
    df00 = df.loc[(df[treatment + "_ego"] == 0) & (df[treatment + "_peers"] == 0)]

    if not (len(df11) > 0 and len(df10) > 0 and len(df01) > 0 and len(df00) > 0):
        raise Exception("Cannot do causal inference, lack of data length: df11", len(df11), ", df10", len(df10), ", df01", len(df01), ", df00", len(df00))

    rfs11 = rfs11.fit(df11.drop([Y], axis=1), df11[Y])
    rfs10 = rfs10.fit(df10.drop([Y], axis=1), df10[Y])
    rfs01 = rfs01.fit(df01.drop([Y], axis=1), df01[Y])
    rfs00 = rfs00.fit(df00.drop([Y], axis=1), df00[Y])

    df_new = df.drop([Y], axis=1)
    ie1 = rfs11.predict(df_new) - rfs01.predict(df_new)
    ie0 = rfs10.predict(df_new) - rfs00.predict(df_new)

    re1 = rfs11.predict(df_new) - rfs10.predict(df_new)
    re0 = rfs01.predict(df_new) - rfs00.predict(df_new)

    oe = rfs11.predict(df_new) - rfs00.predict(df_new)

    return np.mean(list(ie0) + list(ie1)), np.mean(list(re0) + list(re1)), np.mean(oe)


# from causalml.dataset import synthetic_data
# y, X, treatment, tau, b, e = synthetic_data(mode=1, n=40, sigma=.1)
#
# X = np.array([[1]]*len(X))
#
# print(t_learner(X, treatment, y))
# print(x_learner(X, treatment, y))



