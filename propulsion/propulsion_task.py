#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy import stats


# In[2]:


df = pd.read_csv('propulsion.csv')
df.drop('Unnamed: 0', inplace=True, axis=1)
df.head()


# In[3]:


df.shape


# In[4]:


# To predict GT Compressor decay state coefficient
df.columns


# In[5]:


df.rename(columns = {'Lever position (lp) [ ]':'lp','Ship speed (v) [knots]':'speed',
                              'Gas Turbine shaft torque (GTT) [kN m]':'GTT',
                              'Gas Turbine rate of revolutions (GTn) [rpm]':'GTn',
                              'Gas Generator rate of revolutions (GGn) [rpm]':'GGn',
                              'Starboard Propeller Torque (Ts) [kN]':'Ts',
                              'Port Propeller Torque (Tp) [kN]':'Tp',
                              'HP Turbine exit temperature (T48) [C]':'T48',
                              'GT Compressor inlet air temperature (T1) [C]':'T1',
                              'GT Compressor outlet air temperature (T2) [C]':'T2',
                              'HP Turbine exit pressure (P48) [bar]':'P48',
                              'GT Compressor inlet air pressure (P1) [bar]':'P1',
                              'GT Compressor outlet air pressure (P2) [bar]':'P2',
                              'Gas Turbine exhaust gas pressure (Pexh) [bar]':'Pexh',
                              'Turbine Injecton Control (TIC) [%]':'TIC', 'Fuel flow (mf) [kg/s]':'mf',
                              'GT Compressor decay state coefficient.':'Compressor_decay',
                              'GT Turbine decay state coefficient.':'Turbine_decay'},
                               inplace = True)


# In[6]:


df.head()


# In[7]:


df.drop('T1', inplace=True, axis=1)
df.drop('P1', inplace=True, axis=1)


# In[8]:


df.shape


# In[9]:


df.isnull().values.any()


# In[10]:


import seaborn as sns
corr = df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[11]:


corr


# # Applying Various Regression Algorithms

# # Predicting GT Compressor decay state coefficient

# In[12]:


X = df[['lp','speed','GTT','GTn','GGn','Ts','Tp','T48','T2','P48','P2','Pexh','TIC','mf','Turbine_decay']] 
y = df['Compressor_decay']

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[13]:


from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.linear_model import Ridge
import lightgbm as lgb
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import random


# # Random Forest Regression

# In[14]:


# Performing RandomsearchCV on Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
tuned_params = {'n_estimators': [10,20]}  
random_regressor = RandomizedSearchCV(RandomForestRegressor(), tuned_params, n_iter = 2, scoring = 'neg_mean_absolute_error', cv = 3, n_jobs = -1)
random_regressor.fit(X_train, y_train)

# Predicting train and test results (Random forest)
y_train_pred = random_regressor.predict(X_train)
y_test_pred = random_regressor.predict(X_test)

print("Train Results for Random Forest Regressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-Squared: ", r2_score(y_train.values, y_train_pred))

print("\nTest Results for Random Forest Regressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_test, y_test_pred)))
print("R-Squared: ", r2_score(y_test, y_test_pred))


# # XGBoost

# In[15]:


#pip install xgboost


# In[16]:


from numpy import loadtxt
from xgboost import XGBRegressor 


# In[17]:


xgb_reg =  XGBRegressor()
xgb_reg.fit(X_train, y_train)

y_train_pred = xgb_reg.predict(X_train)
y_test_pred = xgb_reg.predict(X_test)

print("Train Results for XGBoost Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-Squared: ", r2_score(y_train.values, y_train_pred))

print("\nTest Results for XGBoost Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_test, y_test_pred)))
print("R-Squared: ", r2_score(y_test, y_test_pred))


# # Predicting GT Turbine decay state coefficient

# In[18]:


X = df[['lp','speed','GTT','GTn','GGn','Ts','Tp','T48','T2','P48','P2','Pexh','TIC','mf','Compressor_decay']] 
y = df['Turbine_decay']

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# # LightGBM

# In[19]:


#pip install lightgbm


# In[20]:


params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'max_depth': 6,
    'num_leaves':10,
    'learning_rate': 0.1,
    'force_col_wise':'true',
    'verbose': 0}
n_estimators = 100


# In[21]:


gbm = lgb.LGBMRegressor(**params)
gbm.fit(X_train, y_train)

y_train_pred = gbm.predict(X_train)
y_test_pred = gbm.predict(X_test)

print("Train Results for LGBMRegressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-Squared: ", r2_score(y_train.values, y_train_pred))

print("\nTest Results for LGBMRegressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_test, y_test_pred)))
print("R-Squared: ", r2_score(y_test, y_test_pred))


# # Ridge Regression

# In[22]:


params = {'alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
ridge_regressor = GridSearchCV(Ridge(), params, cv = 7, scoring = 'neg_mean_absolute_error', n_jobs = -1)
ridge_regressor.fit(X_train, y_train)


# In[23]:


# Predicting train and test results (Ridge regression)
y_train_pred = ridge_regressor.predict(X_train)
y_test_pred = ridge_regressor.predict(X_test)

print("Train Results for Ridge Regressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-Squared: ", r2_score(y_train.values, y_train_pred))

print("\nTest Results for Ridge Regressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_test.values, y_test_pred)))
print("R-Squared: ", r2_score(y_test.values, y_test_pred))

