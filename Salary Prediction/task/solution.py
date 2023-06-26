import os
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape
from scipy import stats as st
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from itertools import combinations

def get_corr_vars(data, value):
    vars = data.corr()[(abs(data.corr()) >= value) & (abs(data.corr()) != 1)]
    cols = data.columns.to_numpy()
    res = set([j for x in vars.notna().to_numpy() for j in cols[x].tolist()])
    return list(res)

def custom_predict(estimator, xtest, ytrain, ytest, mode_bool=False):
    ypreds = estimator.predict(xtest)
    if mode_bool:
        mode_val = st.mode(ytrain, axis=0)
        trimmed_pred = [float(i) if i >= 0 else mode_val.mode[0][0] for i in ypreds]
    else:
        trimmed_pred = [float(i) if i >= 0 else 0 for i in ypreds]
    return round(mape(ytest, trimmed_pred), 5)
def split_train_score(xsub, y, mode_bool):
    X_train, X_test, y_train, y_test = train_test_split(xsub, y, test_size=0.3, random_state=100)
    # fit models
    lr = LinearRegression(fit_intercept=True).fit(X_train, y_train)
    # print(', '.join([str(i) for i in lr.coef_[0]]))
    return custom_predict(lr, X_test, y_train, y_test, mode_bool=mode_bool)




# read data
data = pd.read_csv('C:/Users/roman/PycharmProjects/Salary Prediction/Salary Prediction/task/test/data.csv')
# get correlations > 0.2
corr_vars = get_corr_vars(data, 0.2)
del corr_vars[corr_vars.index("salary")]
# separate features and target
y = np.array(data["salary"]).reshape(-1, 1)
del data["salary"]

# mape_list = []
# for i in list(combinations(corr_vars, 1)):
#     vars = list(i)
#     X_subset = np.array(data.drop(vars, axis=1))
#     mape_list.append(split_train_score(X_subset, y))
#
# for i in list(combinations(corr_vars, 2)):
#     vars = list(i)
#     X_subset = np.array(data.drop(vars, axis=1))
#     mape_list.append(split_train_score(X_subset, y))
#
# print(min(mape_list))
vars = ['experience', 'age']
X_subset = np.array(data.drop(vars, axis=1))

# the best result is achieved trimming negative salary predictions to zero instead of the mode of the training set
print(split_train_score(X_subset, y, False))
#print(split_train_score(X_subset, y, True))