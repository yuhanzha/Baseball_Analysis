#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 21:33:05 2018

@author: yuhanzha
"""
import requests
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

master = pd.read_csv('../input/master.csv')

player = master[['mlb_name','mlb_pos','mlb_team','birth_year','espn_name','espn_id']] 
player.rename(columns={'mlb_name':'Name',
                       'mlb_team':'Team',
                       'mlb_pos':'POS'},inplace =True)

# Get salary dataset           
#sal_url = "https://www.usatoday.com/sports/mlb/salaries/"
#sal_data = pd.read_html(sal_url)
#sal_data = sal_data[0]

sal_data = pd.read_csv('../input/sal_data.csv')

# Remove $
def dollar(col):
    sal_data[col] = sal_data[col].replace('[\$,]', '', regex=True).astype(int)


# Extract Contract lengths

def contract_length(row):
	years = row['Years']
	length = int(years[:years.index("(")].strip())
	return length


# Extract years contract is active
def years_active(row):
	years = row['Years']
	indx_paren = years.index("(")
	indx_close = years.index(")")
	years = str(years[indx_paren+1:indx_close].strip())
	return years


# Extract year contract is start
def year_start (row):
	years = row['Years']
	indx_paren = years.index("(")
	indx_dash = years.find("-")
	year_start = int(years[indx_paren+1:indx_dash])
	return year_start



dollar('Salary')
dollar('Total Value')
dollar('Avg Annual')
sal_data['Contract_Length'] = sal_data.apply (lambda row: contract_length (row),axis=1)	
sal_data['Years_Active'] = sal_data.apply (lambda row: years_active (row),axis=1)
sal_data['Years_Start'] = sal_data.apply (lambda row: year_start (row),axis=1)


dat = pd.merge(sal_data, player, on = ['Name','Team'], how = 'left')
dat.drop('POS_y', axis=1, inplace=True)
dat.rename(columns={'POS_x':'POS'},inplace =True)

war_bat = pd.read_csv('../input/war_daily_bat.csv')
#war_pitch = pd.read_csv('../input/war_daily_pitch.csv')



#  War_bat_salary prediction
bat_sal = pd.merge(dat,  war_bat, left_on="Name", right_on="name_common")
len(bat_sal["Name"].unique())

def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]
    
dif = diff(set(dat["Name"].unique()),set(bat_sal["Name"].unique()))

# Select data before sign an active contract
# Drop players who dosen't have game stats before contract
bat_sal = bat_sal[bat_sal['year_ID']< bat_sal['Years_Start']]
len(bat_sal.Name.unique())



#Avg_Annual = bat_sal[['Name','Team','Avg Annual']]

# Calculate the average all data
temp = bat_sal.groupby(['Name','Team']).mean().reset_index()

# Drop unused variables
d = ['Name','Team','rank','Salary','Total Value','Contract_Length','Years_Start',
     'birth_year','espn_id','mlb_ID','year_ID','stint_ID','salary','teamRpG',
     'oppRpG_rep','pyth_exponent','pyth_exponent_rep','waa_win_perc','waa_win_perc_off',
     'waa_win_perc_def','OPS_plus']
temp2 = temp.drop(d, axis = 1)


# For the final table
d1 = ['Salary','Total Value','Contract_Length','Years_Start',
     'birth_year','espn_id','mlb_ID','year_ID','stint_ID','salary','teamRpG',
     'oppRpG_rep','pyth_exponent','pyth_exponent_rep','waa_win_perc','waa_win_perc_off',
     'waa_win_perc_def','OPS_plus']
temp_name = temp.drop(d1,axis = 1)
temp_name2 = temp_name.dropna()

temp3 = temp2.dropna()


# Feature selection
corr = temp2.corr()
sns.heatmap(corr)

X = temp3.loc[:, temp3.columns != 'Avg Annual']
y = temp3.loc[:, temp3.columns == 'Avg Annual']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

clf = RandomForestRegressor()
clf.fit(X_train, y_train)

# feature importance
importances = clf.feature_importances_

#Sort it
print ("Sorted Feature Importance:")
sorted_feature_importance = sorted(zip(importances, list(X_train)), reverse=True)
print (sorted_feature_importance)


importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

print ("Sorted Feature Importance:")
sorted_feature_importance = sorted(zip(importances, list(X_train)), reverse=True)
print (sorted_feature_importance)

# Select features which importance greater than 0.05
temp4 = temp3[['Avg Annual','oppRpPA_rep','WAR_off','runs_position_p','runs_bat',
              'oppRpG','WAR_rep','Inn']]
corr = temp4.corr()
sns.heatmap(corr)
#a = sns.heatmap(corr)
#corr_plot= a.get_figure()
#corr_plot.savefig("../output/corr_plot.jpg")


# distribution of response variable

sns.distplot(temp4['Avg Annual'], fit = norm)
plt.show()

fig = plt.figure()
res = stats.probplot(temp4['Avg Annual'], plot = plt)
plt.show()



# take log transformation
temp4['Avg Annual'] = np.log1p(temp4['Avg Annual'])
sns.distplot(temp4['Avg Annual'], fit = norm)
plt.show()

# Q-Q plot
fig = plt.figure()
res = stats.probplot(temp4['Avg Annual'], plot = plt)
plt.show()


X = temp4.loc[:, temp4.columns != 'Avg Annual']
y = temp4.loc[:, temp4.columns == 'Avg Annual']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



# Randomforest
clf = RandomForestRegressor(min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



# Tuning parameters for xgboost
"""
xgb_model = XGBRegressor()

parameters = [{'max_depth': [1,2,3],
              'learning_rate': [.001, .01, .05], 
              'n_estimators':[300, 500,1000,3000]}]

grid_search = GridSearchCV(estimator = xgb_model, 
                           param_grid = parameters,
                           scoring ='neg_mean_squared_error',
                           cv = 10,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
best_parameters, score, _ = max(grid_search.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
"""
    
xgb_mod = XGBRegressor(learning_rate =0.05, n_estimators=1000, max_depth=3,
                     min_child_weight=0 ,gamma=0, subsample=0.7,
                     colsample_bytree=0.7,objective= 'reg:linear',
                     nthread=4,scale_pos_weight=1,seed=27, reg_alpha=0.00006)

xgb_model = xgb_mod.fit(X_train, y_train)
y_pred_xgb1 = xgb_model.predict(X_test)



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_xgb1))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_xgb1))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb1)))


# Compare two models
print('Root Mean Squared Error for Random Forest:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Root Mean Squared Error for XGBoost:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb1)))

# Write result
result = {'RMSE':[np.sqrt(metrics.mean_squared_error(y_test, y_pred)),np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb1))]}
result = pd.DataFrame(result, index = ['Random Forest', 'XGBoost'])
#result.to_csv('../output/salary_prediction_rmse.csv')

# Transform results to csv
xgb_model = xgb_mod.fit(X, y)  
y_pred_xgb = xgb_model.predict(X)

#y_pred_xgb2 = np.expm1(xgb_model.predict(X))
#temp4['Prediction'] = y_pred
temp4['Prediction'] = np.expm1(y_pred_xgb)
temp4['Avg Annual'] = np.expm1(temp4['Avg Annual'])

temp4['Value'] = np.where(temp4['Avg Annual']<temp4['Prediction'],'Under Paid','Over Paid')

d = ['Salary','Total Value','Contract_Length','Years_Start',
     'birth_year','espn_id','mlb_ID','year_ID','stint_ID','salary','teamRpG',
     'oppRpG_rep','pyth_exponent','pyth_exponent_rep','waa_win_perc','waa_win_perc_off',
     'waa_win_perc_def','OPS_plus']
temp_name = temp.drop(d,axis = 1)
temp_name2 = temp_name.dropna()

temp_name2['Prediction'] = temp4['Prediction']
temp_name2['Value'] = temp4['Value']

bat_final_xgb = temp_name2[['Name','Team','rank','Avg Annual','Prediction','Value']]

# write to csv
#bat_final_xgb.to_csv('../output/xgb_salary_prediction.csv')



"""
# Pitcher salary prediction
pitch_sal = pd.merge(dat,  war_pitch, left_on="Name", right_on="name_common")
len(pitch_sal["Name"].unique())

pitch_sal = pitch_sal[pitch_sal['year_ID']< pitch_sal['Years_Start']]
len(pitch_sal.Name.unique())

temp = pitch_sal.groupby(['Name','Team']).mean().reset_index()

d1 = ['Salary','Total Value','Contract_Length','Years_Start',
     'birth_year','espn_id','mlb_ID','year_ID','stint_ID','salary','teamRpG',
     'oppRpG_rep','pyth_exponent','waa_win_perc']
temp_name = temp.drop(d1,axis = 1)
temp_name2 = temp_name.dropna()

d = ['Name','Team','rank','Salary','Total Value','Contract_Length','Years_Start',
     'birth_year','espn_id','mlb_ID','year_ID','stint_ID','salary','teamRpG',
     'oppRpG_rep','pyth_exponent','waa_win_perc']

temp2 = temp.drop(d, axis = 1)

temp3 = temp2.dropna()

corr = temp2.corr()
sns.heatmap(corr)

X = temp3.loc[:, temp3.columns != 'Avg Annual']
y = temp3.loc[:, temp3.columns == 'Avg Annual']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


clf = RandomForestRegressor()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

importances = clf.feature_importances_

#Sort it
print ("Sorted Feature Importance:")
sorted_feature_importance = sorted(zip(importances, list(X_train)), reverse=True)
print (sorted_feature_importance)

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
    
temp4 = temp3[['Avg Annual','WAR','WAR_rep','IPouts_start','BIP','pyth_exponent_rep','xRA','xRA_def_pitcher']]
corr = temp4.corr()
sns.heatmap(corr)

X = temp4.loc[:, temp4.columns != 'Avg Annual']
y = temp4.loc[:, temp4.columns == 'Avg Annual']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


clf = RandomForestRegressor()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

temp4['Avg Annual'] = np.expm1(temp4['Avg Annual'])
temp4['Prediction'] = np.expm1(y_pred)

temp4['Value'] = np.where(temp4['Avg Annual']<temp4['Prediction'],'Under Paid','Over Paid')

temp_name2['Prediction'] = temp4['Prediction']    
temp_name2['Value'] = temp4['Value']

pitcher_final = temp_name2[['Name','Team','rank','Avg Annual','Prediction','Value']]
"""
   





















