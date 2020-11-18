#!/usr/bin/env python
# coding: utf-8

# # CAR PRICE PREDICTION

# # IMPORTING LIBRARIES

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import datetime
import matplotlib
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats


# # LOADING AND VISUALISING THE DATASET

# In[2]:


df = pd.read_csv('cars_price.csv')
df.head()


# In[3]:


df.columns


# In[4]:


df.info()


# # ANALYSING TARGET ATTRIBUTE

# In[5]:


df['priceUSD'].describe()


# In[6]:


#skewness and kurtosis
sns.distplot(df['priceUSD'])

print("Skewness: %f" % df['priceUSD'].skew())
print("Kurtosis: %f" % df['priceUSD'].kurt())


# Here we observe that distribution of prices shows a high positive skewness towards left skew > 1. 
# A kurtosis value of 53.8 is extremely high,indicating high presence of outliers in the dataset.

# # ANALYSING RELATIONSHIP WITH NUMERICAL FEATURES 

# In[7]:


def scatter_plot(attrib):
    data = pd.concat([df['priceUSD'], df[attrib]], axis=1)
    data.plot.scatter(x=attrib, y='priceUSD', ylim=(0,195000))


# In[8]:


# mileage vs price
scatter_plot('mileage(kilometers)')


# Here we observe that as the mileage increases the price decreases. Price and Mileage seem to be in an exponential relationship with negative
# exponent. Trend is similar to exponential decay.

# In[9]:


# year vs price
scatter_plot('year')


# Here we observe that price increases in recent years.

# # ANALYSING RELATIONSHIP WITH CATEGORICAL FEATURES

# In[10]:


def box_plot(attrib):
    data = pd.concat([df['priceUSD'], df[attrib]], axis=1)
    f, ax = plt.subplots(figsize=(30, 12))
    fig = sns.boxplot(x=attrib, y="priceUSD", data=data)
    fig.axis(ymin=0, ymax=195000)


# In[11]:


# Let us check how many unique car makers and models do we have in dataset
print("makers :",end="") 
print(len(df.make.unique()))
print("models :",end="") 
print(len(df.model.unique()))


# In[12]:


# car_maker vs price 
box_plot('make')


# Here we observe there is a correlation between Price and Luxury car manufacturers. Also there is a prevalence of low to medium budget cars in the dataset.

# In[13]:


# condition vs price
scatter_plot('condition')


# As expected the price falls if the condition of car is damaged.

# In[14]:


#transmission vs price
scatter_plot('transmission')


# Automatic transmission indeed cost more than the manual ones.

# In[15]:


#segment vs price
sns.catplot(x="segment", y="priceUSD", data=df)


# Here we can observe that segment J,S and E are comparitively costly but almost all other segments have prices uniformly distributed. Here also we can observe presence of some outliers.

# In[16]:


#fuel type distribution
import matplotlib.pyplot as plt
df.groupby('fuel_type')['Unnamed: 0'].nunique().plot(kind = "bar")
plt.show()


# # DATA PREPROCESSING

# In[17]:


# Let us deal with missing data values
df.isnull().sum()


# In[18]:


fuel_group = df.groupby('fuel_type')
fuel_group.describe().head()


# As we observe that count of cars with fuel type electrocar in the dataset is 30 and the volume count for the same is zero so all the missing values in volume column is due to electrocars.we know that volume of fuel consumption for a electrocar is zero so we can replace null values with zero.

# # REMOVING OUTLIERS , CATEGORICAL ENCODING , DEALING WITH MISSING VALUES

# To deal with categorical missing values we use Random Forest Classifier.
# The data preprocessing and encoding part is mentioned in random_forest_classifier file.
# The output is new dataframe with no missing values.

# In[19]:


new_df= pd.read_csv('new.csv')
new_df.head()


# In[20]:


new_df.shape


# In[21]:


new_df.isnull().sum()


# In[22]:


#encode segment attribute
temp_df = pd.DataFrame({
        'segment': [3, 7, 0, 6, 2, 4, 5, 1, 8]
    })
new_df = pd.concat([new_df,pd.get_dummies(new_df['segment'], prefix='segment',drop_first=True)],axis=1)
new_df.drop(['segment'],axis=1, inplace=True)


# chi-square test 

# In[ ]:


from scipy.stats import chi2_contingency 

data = df[['year','priceUSD']]
stat, p, dof, expected = chi2_contingency(data) 
  
# interpret p-value 
alpha = 0.05
print("p value is " + str(p)) 
if p <= alpha: 
    print('Dependent reject null hypothesis variables have a significant relationship') 
else: 
    print('Independent accept null hypothesis variables donot have a significant relationship') 


# In[23]:


corr = new_df.corr()
corr


# Let us visualise the correlation matrix and drop the attributes with weak correlation with the target attribute.

# In[24]:


import seaborn as sns
corr = new_df.corr()
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


# Here dark blue cells indicate strong positive correlation while dark red ones indicate strong negative correlation.

# In[25]:


#drop weak correlations
new_df.drop('Unnamed: 0', inplace = True, axis = 1)
new_df.drop('make', inplace = True, axis = 1)
new_df.drop('condition_with damage', inplace = True, axis = 1)
new_df.drop('condition_with mileage', inplace = True, axis = 1)


# In[26]:


new_df.head()


# Now let us split the dataset into training and testing data and check performance of various machine learning algorithms on data

# In[27]:


from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# In[28]:


X = new_df[['year','mileage(kilometers)','volume(cm3)','segment_1','segment_2','segment_3','segment_4','segment_5','segment_6','segment_7','segment_8','fuel_type_electrocar','fuel_type_petrol','transmission_mechanics','drive_unit_front-wheel drive','drive_unit_part-time four-wheel drive','drive_unit_rear drive']] 
y = new_df['priceUSD']

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


# # Random Forest Regression

# In[29]:


# Performing RandomsearchCV on Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
tuned_params = {'n_estimators': [400,500], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1,2, 4]}  
random_regressor = RandomizedSearchCV(RandomForestRegressor(), tuned_params, n_iter = 10, scoring = 'neg_mean_absolute_error', cv = 5, n_jobs = -1)
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


# # Ridge Regression

# In[30]:


# Performing GridSearchCV on Ridge Regression
from sklearn.linear_model import Ridge
params = {'alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
ridge_regressor = GridSearchCV(Ridge(), params, cv = 7, scoring = 'neg_mean_absolute_error', n_jobs = -1)
ridge_regressor.fit(X_train, y_train)

# Predicting train and test results (Ridge regression)
y_train_pred = ridge_regressor.predict(X_train)
y_test_pred = ridge_regressor.predict(X_test)

print("Train Results for Ridge Regressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-Squared: ", r2_score(y_train.values, y_train_pred))

print("\nTest Results for Ridge Regressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_test.values, y_test_pred)))
print("R-Squared: ", r2_score(y_test.values, y_test_pred))


# # XGBoost

# In[31]:


from sklearn import ensemble
params = {'n_estimators': 400 , 'max_depth':5 ,'learning_rate':0.1, 'criterion': 'mse'}
xgb_reg = ensemble.GradientBoostingRegressor(**params)
xgb_reg.fit(X_train, y_train)

# Predicting train and test results (XGBoost)
y_train_pred = xgb_reg.predict(X_train)
y_test_pred = xgb_reg.predict(X_test)

print("Train Results for XGBoost Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-Squared: ", r2_score(y_train.values, y_train_pred))

print("\nTest Results for XGBoost Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_test.values, y_test_pred)))
print("R-Squared: ", r2_score(y_test.values, y_test_pred))


# Random Forest Model and XGBoost Model provide better accuracy
