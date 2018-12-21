#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:59:30 2018

@author: yuhanzha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import quandl
import seaborn as sns
import math
import statsmodels.api as sm

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


all_salaries = pd.read_csv('../input/baseballdatabank-master/core/Salaries.csv')
all_batting = pd.read_csv('../input/baseballdatabank-master/core/Batting.csv')
all_pitching = pd.read_csv('../input/baseballdatabank-master/core/Pitching.csv')
all_people = pd.read_csv('../input/baseballdatabank-master/core/People.csv')

inflation = pd.read_csv('../input/inflation.csv')
batting = all_batting.sort_values(['playerID', 'yearID'])

# Only seasons where players appeared in more than 100 games were considered
batting = batting[batting['G'] >100]
len(batting.playerID.unique())

# Calculate OBP, SLG and OPS
batting['BA'] = batting['H']/batting['AB']
batting['OBP'] = (batting['H']+batting['BB']+batting['HBP'])/(batting['AB']+batting['BB']+batting['HBP']+batting['SF'])
batting['SLG'] = (batting['H']+batting['2B']+(2*batting['3B'])+(3*batting['HR']))/batting['AB']
batting['OPS'] = batting['OBP']+batting['SLG'] 


# Drop all OPS NA rows
a = batting.dropna(subset = ['OPS'])
len(a.playerID.unique())

# Cumulative games
b = batting.groupby('playerID').G.sum().reset_index()

# Most recent OPS and highest OPS
ops_latest = a.groupby('playerID').last().reset_index()
ops_max = a.groupby('playerID').OPS.max().reset_index()

# 1 stands for 'after peak', and 0 stands for 'before peak'       
ops_latest['PEAK'] = np.where(ops_latest['OPS'] < ops_max['OPS'], 1, 0)

# Use cumulative games
c = pd.merge(ops_latest, b, on = 'playerID', how = 'left')
ops_latest['G']=c['G_y']


# Join with people.csv

bat = pd.merge(ops_latest, all_people, on = 'playerID', how = 'left')
bat = pd.merge(bat, all_salaries, on = ('playerID','yearID'), how = 'left')
bat = bat.dropna(subset = ['salary']).reset_index()
len(bat)

# inflation            
bat = pd.merge(bat,inflation, on = ('yearID'), how = 'left')        
bat['salary_inf'] = bat['salary'] * bat['inflation']

bat = bat[['PEAK','yearID','OPS','G','salary_inf','birthYear','weight','height','bats','throws']]
bat.birthYear = bat.birthYear.astype(int)
bat['age'] = bat['yearID'] - bat['birthYear']
bat = bat.drop(['yearID', 'birthYear'], axis = 1)


###############################################################################

bat['PEAK'].value_counts()

cat_vars = ['bats', 'throws']

# convert categorical variables
for var in cat_vars:
    cat_list = 'var'+'_'+var
    cat_list = pd.get_dummies(bat[var], prefix = var)
    bat1 = bat.join(cat_list)
    bat = bat1
    
bat = bat.drop(['bats','throws'], axis = 1)


X = bat.loc[:, bat.columns != 'PEAK']
y = bat.loc[:, bat.columns == 'PEAK']



# standardizing

#bat_ss = StandardScaler()
columns = X.columns
#X = bat_ss.fit_transform(X.values)
#X = pd.DataFrame(data = X, columns = columns)

# oversampling

os = SMOTE(random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


os_X, os_y = os.fit_sample(X_train,y_train)
os_X = pd.DataFrame(data = os_X, columns = columns)
os_y = pd.DataFrame(data = os_y,columns = ['PEAK'])
# check the numbers of the data
print("length of oversampled data is ", len(os_X))
print("Number of before peak in oversampled data", len(os_y[os_y['PEAK']==0]))
print("Proportion of before peak in oversampled data is", len(os_y[os_y['PEAK']==0])/len(os_X))


# logistic regression


#logit_model = sm.Logit(os_y, os_X)
#result = logit_model.fit()
#print(result.summary2())

log_fit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                             intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                             penalty='l2', random_state=None, solver='liblinear', tol=0.0001)
log_fit.fit(os_X, os_y)

log_fit.score(os_X, os_y)

# Output a pickle file for the model
from sklearn.externals import joblib
joblib.dump(log_fit, '../output/pickle_model2.pkl') 
# Check that the loaded model is the oringinal
log_fit_load = joblib.load('../output/pickle_model2.pkl')
assert log_fit.score(os_X, os_y) == log_fit_load.score(os_X, os_y)

y_pred = log_fit.predict(X_test)


label = {0: 'Wow, this player seems before his peak', 1: 'Unfortunately, this player seems after his peak'}
    #my_prediction  = log_fit.predict(data)[0]
    
my_prediction  = log_fit.predict(X_test)[0]
resfinal = label[my_prediction]
resfinal

# Accuracy
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_fit.score(X_test, y_test)))

# Confusion matrix
log_cm = confusion_matrix(y_test, y_pred)
print(log_cm)

# precision,recall,F-measure and support
logistic_classification_result = classification_report(y_test, y_pred)
print(logistic_classification_result)


# Random Forest

rf_fit = RandomForestClassifier(min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=5, min_samples_split=5,
                                min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=1)
rf_fit.fit(os_X, os_y)

rf_fit.score(os_X, os_y)

y_pred = rf_fit.predict(X_test)

# Accuracy
print('Accuracy of RandomForest classifier on test set: {:.2f}'.format(rf_fit.score(X_test, y_test)))

# precision,recall,F-measure and support
rf_classification_result = classification_report(y_test, y_pred)
print(rf_classification_result)
# Output a pickle file for the model

#joblib.dump(rf_fit, 'pickle_model2.pkl') 
# Check that the loaded model is the oringinal
#rf_fit_load = joblib.load('pickle_model2.pkl')
#assert rf_fit.score(os_X, os_y) == rf_fit_load.score(os_X, os_y)



# Xgboost

# XGBoost tuning parameters
""" 
def xgb_classifier(X_train, X_test, y_train, y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    alg = XGBClassifier(learning_rate=0.1, n_estimators=10, max_depth=5,
                        min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0,
                        objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

    if useTrainCV:
        print("Start Feeding Data")
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(os_X.values, label=os_y.values)
        # xgtest = xgb.DMatrix(X_test.values, label=y_test.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # fit
    print('Start Training')
    alg.fit(os_X, os_y, eval_metric='auc')

    param_test1 = {}
    gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.01, n_estimators=1000, max_depth=5,
                                                    min_child_weight=2, gamma=0.2, subsample=0.8,
                                                    colsample_bytree=0.9,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test1,
                            scoring='f1',
                            n_jobs=4, iid=False, cv=5)
    gsearch1.fit(os_X, os_y)
    print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)

    # make prediction
    print("Start Predicting")
    predictions = alg.predict(X_test)
    pred_proba = alg.predict_proba(X_test)[:, 1]

    # results
    print("\nabout this model")
    print("accuracy : %.4g" % metrics.accuracy_score(y_test, predictions))
    print("AUC: %f" % metrics.roc_auc_score(y_test, pred_proba))
    print("F1 Score: %f" % metrics.f1_score(y_test, predictions))
    print("recall" % metrics.recall_score(y_test,predictions))

    feat_imp = alg.feature_importances_
    feat = os_X.columns.tolist()
    # clf.best_estimator_.booster().get_fscore()
    res_df = pd.DataFrame({'Features': feat, 'Importance': feat_imp}).sort_values(by='Importance', ascending=False)
    res_df.plot('Features', 'Importance', kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()
    print(res_df)
    print(res_df["Features"].tolist())

xgb_classifier(os_X, X_test, os_y, y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=50)
"""

xgb_mod = XGBClassifier(learning_rate=0.1, n_estimators=10, max_depth=5,
                        min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0,
                        objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

xgb_mod.fit(os_X, os_y, eval_metric='auc')
predictions = xgb_mod.predict(X_test)

xgb_classification_result = classification_report(y_test, predictions)
print(xgb_classification_result)

#joblib.dump(xgb_mod, '../output/pickle_model.pkl') 


