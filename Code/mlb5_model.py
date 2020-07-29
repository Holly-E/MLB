# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:50:09 2019

@author: Erick
"""
# Speed at .28


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.ensemble import VotingRegressor

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

#%%
#df =  pd.read_csv('C:/Baseballcloud/MLB Data/Clusters/L_pitches_w_zone.csv')
dfr =  pd.read_csv('C:/Baseballcloud/MLB Data/Clusters/R_pitches_w_zone.csv')

#df = dfl.append(dfr)

#%%
df = dfr[pd.notnull(dfr['estimated_woba_using_speedangle'])]
#%%
cols = list(df.columns)


#%%
# pfxx = horz break & pfxz = indvert
feats = ['BatterSide', 'HorzBreak', 'InducedVertBreak', 'PlateLocHeight', 'PlateLocSide','RelSpeed', 'Spin Efficiency','True Spin (rpm)','ay', 'release_pos_x', 'release_pos_y', 'release_pos_z',  'balls', 'strikes', 'ExitSpeed', 'Elevation']
drop = [x for x in cols if x not in feats]
df.drop(drop, axis = 1, inplace = True)

#%%

df.drop(drop, axis = 1, inplace = True)
#%%
df.dropna(inplace = True)
df['BatterSide'] = np.where(df['BatterSide'] == 'R', 1, 0)
#df.to_csv('C:/Baseballcloud/MLB Data/Clusters/L_MLBxwOBApitchdata.csv', index = False)

#%%
predictors = [c for c in df.columns if c not in ['ExitSpeed']]
xTrain, xTest, yTrain, yTest = train_test_split(df[predictors], df['ExitSpeed'], test_size = 0.2, random_state = 0)

#%%
num_feats = 10
#  Pearson's Correlation
def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature
cor_support, cor_feature = cor_selector(xTrain, yTrain,num_feats)

#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()



# f_regression selectKBest

X_norm = MinMaxScaler().fit_transform(xTrain)
chi_selector = SelectKBest(f_regression, k=num_feats)
chi_selector.fit(X_norm, yTrain)
chi_support = chi_selector.get_support()
chi_feature = xTrain.loc[:,chi_support].columns.tolist()


# Recursive feature elimination with linear regression
rfe_selector = RFE(estimator=LinearRegression(n_jobs= -1), n_features_to_select=num_feats, step=10, verbose=1)
rfe_selector.fit(X_norm, yTrain)
rfe_support = rfe_selector.get_support()
rfe_feature = xTrain.loc[:,rfe_support].columns.tolist()


# Meta-transformer for selecting features based on importance weights
embeded_lr_selector = SelectFromModel(LinearRegression(n_jobs = -1), max_features=num_feats)
embeded_lr_selector.fit(X_norm, yTrain)
embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = xTrain.loc[:,embeded_lr_support].columns.tolist()


# Random forest’s feature importance
embeded_rf_selector = SelectFromModel(RandomForestRegressor(n_estimators=50, n_jobs = -1), max_features=num_feats)
embeded_rf_selector.fit(xTrain, yTrain)
embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = xTrain.loc[:,embeded_rf_support].columns.tolist()

feature_name = xTrain.columns.tolist()


# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'KBest':chi_support, 'RFE':rfe_support, 'Embeded_lr':embeded_lr_support,
                                    'Random Forest':embeded_rf_support}) #, 'LightGBM':embeded_lgb_support
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)

#%%
feature_selection_df.to_csv('C:/Baseballcloud/MLB Data/Clusters/L_feature_selection.csv', index = False)
#feature_selection_df = pd.read_csv('feature_selection.csv')
#print(feature_selection_df.columns)

# Only keep features in top 10 for at least 2 of the 5 methods
feats = []
for ind, row in feature_selection_df.iterrows():
    if row['Total']>= 3:
        feats.append(row['Feature'])
        
#%%
        
# THIS IS THE SECTION TO CREATE PREDICTIVE MODEL AND TUNE HYPERPARAMETERS
scaler = MinMaxScaler()
#df.columns = df.columns.astype(str)
#train_rel = df[feats]
X_norm = scaler.fit_transform(df)
xTrain, xTest, yTrain, yTest = train_test_split(X_norm, df['Exit Speed'], test_size = 0.2, random_state = 0)

#%%
# Create a ridge reg model
ridge = Ridge()
# Create a dictionary of all values to test
params_ridge = {'alpha': np.logspace(-6, 6, 13)}
# Use gridsearch to test all values for n_neighbors
ridge_gs = GridSearchCV(ridge, params_ridge, cv=3, verbose = 2, n_jobs = -1)
# Fit model to training data
ridge_gs.fit(xTrain, yTrain)
# Save best model
ridge_best = ridge_gs.best_estimator_
# Check best n_neigbors value
print(ridge_gs.best_params_)
print('ridge: {}'.format(ridge_best.score(xTest, yTest)))
#%%
# Create a random forest model
rf = RandomForestRegressor()
# Create a dictionary of all values we want to test for n_estimators
params_rf = {'n_estimators': [200, 500],
             'max_depth': [5,10],
             }
# Use gridsearch to test all values for n_estimators
rf_gs = GridSearchCV(rf, params_rf, cv=3, verbose = 2, n_jobs = -1)
# Fit model to training data
rf_gs.fit(xTrain, yTrain)
# Save best model
rf_best = rf_gs.best_estimator_
# Check best n_estimators value
print(rf_gs.best_params_)

print('rf: {}'.format(rf_best.score(xTest, yTest)))

#%%
# Create a new linear regression model
lin_reg = LinearRegression()
# Fit the model to the training data
lin_reg.fit(xTrain, yTrain)
# Save best model
lin_best = lin_reg

print('lin_reg: {}'.format(lin_best.score(xTest, yTest)))

#%%
# Create a lasso reg 
lass = Lasso()
params_lass = {'alpha': np.logspace(-6, 6, 13)}
# Use gridsearch to test all values for n_estimators
lass_gs = GridSearchCV(lass, params_lass, cv=3, verbose = 2, n_jobs = -1)
# Fit the model to the training data
lass_gs.fit(xTrain, yTrain)
# Save best model
lass_best = lass_gs.best_estimator_
# Check best n_estimators value
print(lass_gs.best_params_)

print('lasso: {}'.format(lass_best.score(xTest, yTest)))
