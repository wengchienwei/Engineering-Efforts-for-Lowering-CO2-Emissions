#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis,skew
import random as rd


# In[6]:


print(pd.__version__)


# In[7]:


path_data = r"C:\\Engineering Efforts for Lowering CO2 Emissions\co2_emissions.csv"


# In[8]:


df = pd.read_csv(path_data, index_col='id') 
df.sample(10)


# # 1. Data Understanding
# 
# ## 1.1 Overview
# 
# ##### 1. We have **16** different variables(series) with **3** different data types.
# 
# ##### 2. We found that we have missing values in **5** variables.
# 
# ## 1.2 Variables Explanation
# 
# ##### 1. brand: We have data of **49** different brands, e.g. Auston Martin, audi, etc.
# 
# ##### 2. commercial_designation: e.g. id: 1, we have "159 1750 Tbi (200ch)" as commercial designation.
# 
# TBi - stands for turbo petrol engine with direct injection, so this is 1.75 liter turbo engine with 200hp.
# 
# 159 - stands for highest sport pack that was available on 159.
# 
# ##### 3. utac_model: The testing UTAC model of the vehicle.
# 
# ##### 4. fuel_type: GO(Gazole), ES(Essence)
# 
# ##### 8. transmission: A(Automatic) or M(Manual) and number of gears.
# 
# ##### 15. range: the distance a vehicle can travel.
# 
# ##### 16. co2_gkm: the CO2 emissions in grams per km. **This is the explained variable that we concern.**

# In[9]:


columns_headers = list(df.columns)
columns_headers


# In[10]:


unique_count = df['brand'].nunique()
print("Number of unique variables in 'brand' column:", unique_count)


# In[11]:


df.shape


# In[12]:


df.dtypes


# In[13]:


df.info()


# ## 1.3 Univariate Statistics and Analysis Focused on Explained Variable

# In[14]:


df.describe()


# Analysis of the data
# 1. The **mean** of the CO2 emissions in grams per km is **172.136** with a somewhat high **std** of **51.3**. The **skewness** is **1.047(>0)** which means the distribution is skewed to the right. It tends to have cars with high CO2 emission as outliers. The **kurtosis** is **2.029** which means the distribution has heavier tails compared to normal distribution. High CO2 emissions from some models may be a normal market phenomenon.
# 2. The maximum of **administrative_power** and **maximum_power_kW** is far away from the mean, so we need to exam if there are abonrmal values or typos.

# In[15]:


print(df['co2_gkm'].describe())

co2_gkm_skew = skew(df['co2_gkm'])
print('\nThe skewness of co2_gkm: ', co2_gkm_skew)

co2_gkm_kurtosis = kurtosis(df['co2_gkm'])
print('The kurtosis of co2_gkm: ',co2_gkm_kurtosis)


# ## *Revise 1* 
# **After the presentation, we examed whether the variables in string columns (range, body type, and transmission) will contribute to the target variable**

# ## 1.4 Univariate Statistics and Examination of Category Variable

# ### a. Exam the mean of CO2 emissions for each variable in **range**

# In[16]:


# Calculate the mean of 'co2_gkm'
overall_mean_co2 = df['co2_gkm'].mean()
mean_co2_gkm_range = df.groupby('range')['co2_gkm'].mean()
mean_co2_gkm_range = mean_co2_gkm_range.sort_values(ascending=False)


# In[17]:


# Plotting the horizontal bar plot
plt.figure(figsize=(10, 6))
bars = plt.barh(mean_co2_gkm_range.index, mean_co2_gkm_range, color='skyblue', label='Mean CO2 Emissions per Range')
plt.axvline(x=overall_mean_co2, color='red', linestyle='--', label='Overall Mean CO2 Emissions')

# Add counts to the right-hand side of each bar
for bar, count in zip(bars, df.groupby('range').size().sort_index()):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{count}', va='center', fontsize=0.8*plt.rcParams['font.size'])

plt.title('Comparison of Mean CO2 Emissions per Range with Overall Mean')
plt.xlabel('Mean CO2 Emissions (g/km)')
plt.ylabel('Range')
plt.legend()
plt.tight_layout()
plt.show()


# ### b. Exam the mean of CO2 emissions for each variable in **body_type**

# In[18]:


# Calculate the mean of 'co2_gkm'
mean_co2_gkm_body_type = df.groupby('body_type')['co2_gkm'].mean()
mean_co2_gkm_body_type = mean_co2_gkm_body_type.sort_values(ascending=False)
overall_mean_co2 = df['co2_gkm'].mean()


# In[19]:


# Plotting the horizontal bar plot
plt.figure(figsize=(10, 6))
bars = plt.barh(mean_co2_gkm_body_type.index, mean_co2_gkm_body_type, color='skyblue', label='Mean CO2 Emissions per Body type')
plt.axvline(x=overall_mean_co2, color='red', linestyle='--', label='Overall Mean CO2 Emissions')

# Add counts to the right-hand side of each bar for body type
for bar, count in zip(bars, df.groupby('body_type').size().sort_index()):
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{count}', va='center', fontsize=0.8*plt.rcParams['font.size'])

plt.title('Comparison of Mean CO2 Emissions per Body type with Overall Mean')
plt.xlabel('Mean CO2 Emissions (g/km)')
plt.ylabel('Body type')
plt.legend()
plt.tight_layout()
plt.show()


overall_std_co2 = df['co2_gkm'].std()
print('\nOverall Standard Deviation CO2 Emissions: ', overall_std_co2)


# ### c. Exam the mean of CO2 emissions for each variable in **transmission**

# In[20]:


# Calculate the mean of 'co2_gkm'
mean_co2_gkm_transmission = df.groupby('transmission')['co2_gkm'].mean()
mean_co2_gkm_transmission = mean_co2_gkm_transmission.sort_values(ascending=False)
overall_mean_co2 = df['co2_gkm'].mean()


# In[21]:


# Plotting the horizontal bar plot
plt.figure(figsize=(10, 6))
bars = plt.barh(mean_co2_gkm_transmission.index, mean_co2_gkm_transmission, color='skyblue', label='Mean CO2 Emissions per Transmission')
plt.axvline(x=overall_mean_co2, color='red', linestyle='--', label='Overall Mean CO2 Emissions')

# Add counts to the right-hand side of each bar for body type
for bar, count in zip(bars, df.groupby('transmission').size().sort_index()):
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{count}', va='center', fontsize=0.8*plt.rcParams['font.size'])


plt.title('Comparison of Mean CO2 Emissions per Transmission with Overall Mean')
plt.xlabel('Mean CO2 Emissions (g/km)')
plt.ylabel('Transmission')
plt.legend()
plt.tight_layout()
plt.show()


overall_std_co2 = df['co2_gkm'].std()
print('\nOverall Standard Deviation CO2 Emissions: ', overall_std_co2)


# We can see that the means of CO2 emissions among each variable do not change much, especially when we focus on the top 3 counts variable for each variable. Besides we found that the differences between the means of CO2 emissions per variable and the overall mean of CO2 emissions are within the overall standard deviation.
# 
# 
# Therefore, the category variables do not have a great impact on the target variable, and we will ignore the category variables when we train our regression models.

# # 2. Data Processing and Cleaning 
# 
# We found that **administrative_power** and **maximum_power_kW** have large maximum values away from the mean and high standard deviation, so we will focus on examining the data points in these two variables.

# ## 2.1 Dealing with The Outliers

#  ### a. Examination of **administrative_power**

# In[22]:


plt.figure(figsize=(7, 5))
plt.boxplot(df.administrative_power, notch=True)
plt.title("Numeric variable - the administrative power of the vehicle", 
          size=14)
plt.ylabel("Administrative Power")
plt.xticks([1], [""])
plt.grid(alpha=0.4)


# In[23]:


df.loc[df['administrative_power'] >= 65.0, 
       ['brand', 'administrative_power', 'maximum_power_kW', 'mixed_fuel_consumption', 'co2_gkm']]


# Despite these 6 outliers (**administrative_power** >= 65) having relatively high administrative power, they also exhibit large maximum power and mixed fuel consumption, so we decide to **keep these 6 extreme outliers unchanged**.

# ### b. Examination of **maximum_power_kW**

# In[24]:


plt.figure(figsize=(7, 5))
plt.boxplot(df.maximum_power_kW, notch=True)
plt.title("Numeric variable - the maximum power of the vehicle in kW", 
          size=14)
plt.ylabel("maximum power in kW")
plt.xticks([1], [""])
plt.grid(alpha=0.4)


# In[25]:


df.loc[df['maximum_power_kW'] >= 700.0, 
       ['brand', 'fuel_type', 'administrative_power', 'maximum_power_kW', 'mixed_fuel_consumption', 'co2_gkm']]


# In[26]:


print('Median of administrative_power: ', df['administrative_power'].median())
print('Median of maximum_power_kW: ', df['maximum_power_kW'].median())
print('Median of mixed_fuel_consumption: ', df['mixed_fuel_consumption'].median())
print('Median of co2_gkm: ', df['co2_gkm'].median())


# These vehicle models have relatively high **maximum_power_kW**, while their other variables stay in normal range. They might be abnormal values. However, there are so many *MERCEDES-BENZ* vehicles that show up in the table. Therefore, we only replace the values of vehicles with the median of 'maximum_power_kW' for those that aren't *MERCEDES-BENZ*.

# ## *Revise 2* 
# **After the presentation, we replaced the missing values with median instead of mean**

# In[27]:


median_maximum_power = df['maximum_power_kW'].median()
df.loc[(df['maximum_power_kW'] >= 700.0) & (df['brand'] != 'MERCEDES-BENZ'), 'maximum_power_kW'] = median_maximum_power


# ## 2.2 Dealing with Missing Values

# In[28]:


nb_missing_values = df.isna().sum()
nb_missing_values


# ## a. Finding Missing Value

# In[29]:


import seaborn as sns

plt.title('Missing Value',fontweight='bold')
ax = sns.heatmap(df.isna().sum().to_frame(),annot=True,fmt='d',cmap='vlag')
ax.set_xlabel('Amount Missing')
plt.show()


# ## b. Filling Missing Value of Float with The Median Method

# ## *Revise 3* 
# **After the presentation, we replaced the missing values with median instead of mean**

# In[30]:


df['urban_fuel_consumption'].fillna(df['urban_fuel_consumption'].median(), inplace=True)


# In[31]:


df['extra_urban_fuel_consumption'].fillna(df['extra_urban_fuel_consumption'].median(), inplace=True)


# In[32]:


df['min_euro_empty_weight'].fillna(df['min_euro_empty_weight'].median(), inplace=True)


# In[33]:


df.isna().sum()


# ## c. Filling The Missing Values of String with The Mode

# In[34]:


mode_value_body_type = df['body_type'].mode()[0]
mode_value_body_type


# In[35]:


df['body_type'] = df['body_type'].fillna(mode_value_body_type)


# In[36]:


mode_value_range = df['range'].mode()[0]
mode_value_range


# In[37]:


df['range'] = df['range'].fillna(mode_value_range)


# In[38]:


df.isna().sum()


# ## d. Transforming data types

# Convert **maximum_power_kW** and **min_euro_empty_weight** into integer

# In[39]:


df['maximum_power_kW'] = df['maximum_power_kW'].astype(int)


# In[40]:


df['min_euro_empty_weight'] = df['min_euro_empty_weight'].astype(int)


# In[41]:


df_object = df.select_dtypes(include='object')

for col in df_object.columns:
    print(df[col].value_counts())
    print('-------------------')


# Convert **hybrid** into boolean

# In[42]:


list_to_boolean = ['hybrid']

def transform_hybrid_type(hybrid):
    if  hybrid == 'oui':
        output = True
    else:
        output = False
    return output

df['hybrid'] = df['hybrid'].apply(transform_hybrid_type)


# In[43]:


df.hybrid


# # 3. Visualization

# In[44]:


plt.figure(figsize=(7, 5))
plt.hist(df.co2_gkm, density=True, bins=250, alpha=0.4)
plt.title(" Distribution of 'CO2 emissions in grams per km'", size=14)
plt.ylabel("Repartition (%)")
plt.xlabel("CO2  emissions")
plt.grid(alpha=0.4)
plt.show()


# In[45]:


plt.figure(figsize=(7, 5))
plt.boxplot(df.co2_gkm, notch=True)
plt.title("Integer variable - CO2  emissions in grams per km", 
          size=14)
plt.ylabel("CO2  emissions")
plt.xticks([1], [""])
plt.grid(alpha=0.4)


# In[46]:


df.loc[df['co2_gkm'] >= 500]


# In[47]:


df.loc[(df['brand'] == 'ASTON MARTIN') & (df['co2_gkm'] >= 350), 
       ['brand', 'administrative_power', 'maximum_power_kW', 'mixed_fuel_consumption', 'max_euro_empty_weight', 'co2_gkm']]


# Although there are two extremely high values in **co2_gkm**, we compared these two data with **maximum_power_kW** and **co2_gkm** variables. We found that *ASTON MARTIN* tend to have relatively high values in terms of vehicle model in these two variables. Therefore, they seem not to be the outliers.

# # 4. Holdout and Encoding

# In[48]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
import warnings; warnings.simplefilter('ignore')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


# In[49]:


df.info()


# ## 4.1 One-hot Encoding

# In[189]:


# Initialize the OneHotEncoder
ohe = OneHotEncoder()
# create a list with variables you would like to encode
list_encode = ['brand', 'commercial_designation','utac_model', 'fuel_type', 'transmission', 'body_type', 'range']
# Fit and transform the gender, work_type and smoking_status columns
ohe_df = ohe.fit_transform(df[list_encode])
# Create a dataframe for the one-hot-encoded features
df_ohe = pd.DataFrame(ohe_df.toarray(), columns=[col  for array in ohe.categories_ for col in array], index=df.index)
# Concatenate the one-hot-encoded features with the original data
df2 = pd.concat([df[df.columns[~df.columns.isin(list_encode)]], df_ohe], axis=1)


# ## 4.2 Holdout

# In[190]:


X = df2.drop('co2_gkm', axis=1)
y = df2['co2_gkm']


# In[191]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=50)


# In[192]:


X_train.head(3)


# In[193]:


X_test.head(3)


# ## 4.3 Convert One-hot Encoding Variables into Boolean Variables

# In[194]:


list_df = [X_train, X_test]
for df2 in list_df:
    for col in [col  for array in ohe.categories_ for col in array]:
        df2[col] = df2[col].astype('bool')


# In[195]:


X_train.dtypes


# In[196]:


X_test.dtypes


# # 5. Scale The Data

# In[199]:


stsc = StandardScaler()
# Create a list of numerical columns
numerical_cols = ['administrative_power', 'maximum_power_kW', 'urban_fuel_consumption',
                     'extra_urban_fuel_consumption', 'mixed_fuel_consumption',
                     'min_euro_empty_weight', 'max_euro_empty_weight']
# Fit and transform the numerical columns
X_train[numerical_cols] = stsc.fit_transform(X_train[numerical_cols])
#transform the numerical columns
X_test[numerical_cols] = stsc.transform(X_test[numerical_cols])


# In[200]:


X_train.head()


# # 6. K-Nearest Neighbors (KNN) & K-fold Cross-Validation 
# The model exams the performance of the prediction of training dataset with all the meaningful variables.

# In[208]:


features = ['administrative_power', 'maximum_power_kW', 'urban_fuel_consumption',
            'extra_urban_fuel_consumption', 'mixed_fuel_consumption', 'min_euro_empty_weight',
            'max_euro_empty_weight', 'hybrid']

X_train_features = X_train[features]
X_test_features = X_test[features]


# In[209]:


k = 5
knn_reg = KNeighborsRegressor(n_neighbors=k)

# Train the K-neighbors model
knn_reg.fit(X_train_features, y_train)

# Predict on test set
y_pred_knn = knn_reg.predict(X_test_features)

# Calculate MSE for K-neighbors
mse_knn = mean_squared_error(y_test, y_pred_knn)
print("Mean Squared Error (MSE) for K-nearest neighbors:", round(mse_knn, 2))

# Calculate R^2 for K-neighbors
r2_knn = r2_score(y_test, y_pred_knn)

# Calculate adjusted R^2 for K-neighbors
n = len(y_test)
p = len(features)
adj_r2_knn = 1 - ((1 - r2_knn) * (n - 1) / (n - p - 1))
print("Adjusted R^2 score for K-nearest neighbors:", round(adj_r2_knn, 2))


# In[210]:


# K-fold cross-validation
cv_k = 5
knn_reg_cv = KNeighborsRegressor(n_neighbors=5)


# Run K-fold cross-validation
cv_scores = cross_val_score(knn_reg_cv, X_train_features, y_train, cv=cv_k, scoring='r2')

# Calculate MSE for K-fold cross-validation
mse_folds = -cross_val_score(knn_reg_cv, X_train_features, y_train, cv=cv_k, scoring='neg_mean_squared_error')


print("Mean Squared Error (MSE) for K-folds:", mse_folds)
print("Mean of MSE for K-folds:", round(np.mean(mse_folds), 2))
print("Standard Deviation of MSE:", round(np.std(mse_folds), 2))


# Calculate adjusted R^2 for K-fold cross-validation
r2_cv = cv_scores.mean()
adj_r2_cv = 1 - ((1 - r2_cv) * (n - 1) / (n - p - 1))
print("Adjusted R^2 score for K-fold cross-validation:", round(adj_r2_cv, 2))


# In[211]:


import matplotlib.pyplot as plt

# Data
mse_knn = 30.85
adj_r2_knn = 0.99

mse_kfolds = [65.86117949, 56.6025641, 33.67758974, 28.03102564, 62.52426187]
mean_mse_kfolds = 49.34
std_mse_kfolds = 15.48
adj_r2_kfolds = 0.98

# Plotting
plt.figure(figsize=(10, 6))

# Plot MSE
plt.subplot(1, 2, 1)
plt.bar(["KNN", "K-Folds"], [mse_knn, mean_mse_kfolds], yerr=[0, std_mse_kfolds], color=['blue', 'orange'], alpha=0.7, capsize=10)
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Mean Squared Error Comparison')
plt.grid(axis='y')

# Plot Adjusted R^2
plt.subplot(1, 2, 2)
plt.bar(["KNN", "K-Folds"], [adj_r2_knn, adj_r2_kfolds], color=['blue', 'orange'], alpha=0.7)
plt.ylabel('Adjusted R^2')
plt.title('Adjusted R^2 Comparison')
plt.ylim(0.95, 1)  # Limiting y-axis for better visualization
plt.grid(axis='y')

plt.tight_layout()
plt.show()


# # 7. Grid Search

# ## 7.1 Grid search 1
# (Using hyperparameter tuning with k-fold)

# In[212]:


k_grid = np.arange(1, 10, 1)
output_mean = []
for k in k_grid:
    knn = KNeighborsRegressor(n_neighbors=k)
    results_k = cross_validate(knn, X_train_features, y_train, cv=5, 
                               scoring ="neg_mean_squared_error")
    mse_k = -np.mean(results_k["test_score"])
    output_mean.append(mse_k)


# In[213]:


output_mean


# In[214]:


plt.figure(figsize=(10, 6))
plt.plot(k_grid, output_mean, marker='o', linestyle='-')
plt.title('Which k should we choose by cosidering the MSE on the train set?')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Squared Error (MSE)')
plt.xticks(np.arange(0, 10, step=1))
plt.grid(True)
plt.show()


# ## 7.2 Grid search 2
# (Deciding the best parameters for the **'weights'** and **'metric'** )

# In[215]:


# Define the parameter grid
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 15],  # can be adjusted
    'weights': ['uniform', 'distance'],
    'metric' : ['euclidean', 'manhattan']
}

# Perform grid search with k-fold cross-validation
grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train_features, y_train)


# In[216]:


# Get the best model
best_knn_reg = grid_search.best_estimator_

# Predict on the test set using the best model
y_pred_grid = best_knn_reg.predict(X_test_features)

# Calculate MSE for the best model
mse_grid = mean_squared_error(y_test, y_pred_grid)
print("Mean Squared Error (MSE) for grid search with k-fold:", round(mse_grid, 2))

# Print the best hyperparameters found by the grid search
print("Best hyperparameters:", grid_search.best_params_)


# # 8. Model Selection - Evaluating The Importance of Variables

# ## 8.1 K-Nearest Neighbors (KNN) Regression Model
# Based on the best parameters from the grid search, we run the KNN regression model with K-fold to rank feature importance scores by permuting the current feature and observing the change in Mean Squared Error.

# In[528]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np


# In[218]:


# Best hyperparameters from grid search
best_hyperparameters = {'metric': 'manhattan', 'n_neighbors': 5, 'weights': 'distance'}

# Numeric features
numeric_features = ['administrative_power', 'maximum_power_kW', 'urban_fuel_consumption',
                    'extra_urban_fuel_consumption', 'mixed_fuel_consumption', 'min_euro_empty_weight',
                    'max_euro_empty_weight']

X_train_numeric_features = X_train[numeric_features]
y_train = y_train


# In[219]:


# Initialize K-nearest neighbors regressor with best hyperparameters
knn_reg = KNeighborsRegressor(**best_hyperparameters)

# Define the number of folds for cross-validation
k = 3
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Initialize an array to store feature importance scores
feature_importance_scores = np.zeros(len(numeric_features))

# Calculate base MSE without permuting features
mse_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_numeric_features, y_train)):
    X_train_fold, X_val_fold = X_train_numeric_features.iloc[train_idx], X_train_numeric_features.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    knn_reg.fit(X_train_fold, y_train_fold)
    y_pred_val = knn_reg.predict(X_val_fold)
    mse_scores.append(mean_squared_error(y_val_fold, y_pred_val))

# Iterate over each feature
for i, feature in enumerate(numeric_features):
    # Initialize an array to store MSE scores with permuted feature
    mse_permuted = np.zeros(k)
    
    # Iterate over each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_numeric_features, y_train)):
        X_train_fold, X_val_fold = X_train_numeric_features.iloc[train_idx], X_train_numeric_features.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Copy the training data and randomly permute the current feature
        X_train_permuted = X_train_fold.copy()
        X_train_permuted[feature] = np.random.permutation(X_train_permuted[feature])
        
        # Train a K-nearest neighbors regression model
        knn_reg.fit(X_train_permuted, y_train_fold)
        
        # Predict on the validation set
        y_pred_val = knn_reg.predict(X_val_fold)
        
        # Compute MSE for this fold with permuted feature
        mse_permuted[fold] = mean_squared_error(y_val_fold, y_pred_val)
    
    # Calculate the change in MSE caused by permuting the current feature
    feature_importance_scores[i] = np.mean(mse_permuted) - np.mean(mse_scores)

# Create a dictionary mapping feature names to their importance scores
feature_importance_dict = dict(zip(numeric_features, feature_importance_scores))

# Sort features by their importance scores in descending order
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Print the ranking of key variables
print("Ranking of key variables based on feature importance scores:")
for rank, (numeric_feature, importance_score) in enumerate(sorted_features, start=1):
    print(f"{rank}. {numeric_feature}: {importance_score}")


# In[220]:


import matplotlib.pyplot as plt

# Extracting sorted feature names and importance scores
sorted_numeric_features = [feature[0] for feature in sorted_features]
sorted_importance_scores = [feature[1] for feature in sorted_features]

# Plotting feature importance scores
plt.figure(figsize=(10, 6))
bars = plt.barh(sorted_numeric_features, sorted_importance_scores, color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Numeric Feature')
plt.title('Feature Importance Scores (KNN Regression)')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important features at the top

# Add importance scores on the right-hand side of each bar
for bar, score in zip(bars, sorted_importance_scores):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{score:.2f}', 
             va='center', ha='left', fontsize=0.75*plt.rcParams['font.size'])

plt.show()


# ## 8.2 Linear Regression
# The coefficients represent the change in the CO2 emissions for an one-unit change in the corresponding variable, assuming all other variables are fixed.

# In[221]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet


# In[222]:


numeric_features = ['administrative_power', 'maximum_power_kW', 'urban_fuel_consumption',
            'extra_urban_fuel_consumption', 'mixed_fuel_consumption', 'min_euro_empty_weight',
            'max_euro_empty_weight']

X_train_numeric_features = X_train[numeric_features]
y_train = y_train


# In[223]:


# Initialize Linear Regression model
lin_reg = LinearRegression()

# Define the number of folds for cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Initialize an array to store feature importance scores
feature_importance_scores = np.zeros(len(numeric_features))  # Use length of numeric_features

# Iterate over each fold
for train_idx, val_idx in kf.split(X_train_numeric_features, y_train):
    X_train_fold, X_val_fold = X_train_numeric_features.iloc[train_idx], X_train_numeric_features.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train a linear regression model
    lin_reg.fit(X_train_fold, y_train_fold)
    
    # Compute absolute feature importance scores (absolute coefficients)
    feature_importance_scores += np.abs(lin_reg.coef_)

# Average feature importance scores across all folds
average_feature_importance_scores = feature_importance_scores / k

# Create a dictionary mapping feature names to their average importance scores
feature_importance_dict = dict(zip(numeric_features, average_feature_importance_scores))

# Sort features by their average importance scores in descending order
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Print the ranking of key variables
print("Ranking of key variables:")
for rank, (numeric_feature, importance_score) in enumerate(sorted_features, start=1):
    print(f"{rank}. {numeric_feature}: {importance_score}")


# In[224]:


import matplotlib.pyplot as plt

# Extracting sorted feature names and importance scores
sorted_numeric_features = [feature[0] for feature in sorted_features]
sorted_importance_scores = [feature[1] for feature in sorted_features]

# Plotting feature importance scores
plt.figure(figsize=(10, 6))
bars = plt.barh(sorted_numeric_features, sorted_importance_scores, color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Numeric Feature')
plt.title('Feature Importance Scores (Linear Regression)')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important features at the top

# Add importance scores on the right-hand side of each bar
for bar, score in zip(bars, sorted_importance_scores):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{score:.2f}', 
             va='center', ha='left', fontsize=0.75*plt.rcParams['font.size'])

plt.show()


# ## 8.3 Elastic-Net
# To avoid overfitting, we used the **Elastic-Net**, which combined L1 and L2 regularization, allowing to penalize the residuals.

# In[225]:


# Initialize Elastic Net model
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)  # You can adjust alpha and l1_ratio as needed

# Define the number of folds for cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Initialize an array to store feature importance scores
feature_importance_scores = np.zeros(len(numeric_features))  # Use length of numeric_features

# Iterate over each fold
for train_idx, val_idx in kf.split(X_train_numeric_features, y_train):
    X_train_fold, X_val_fold = X_train_numeric_features.iloc[train_idx], X_train_numeric_features.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train an Elastic Net model
    elastic_net.fit(X_train_fold, y_train_fold)
    
    # Compute absolute feature importance scores (absolute coefficients)
    feature_importance_scores += np.abs(elastic_net.coef_)

# Average feature importance scores across all folds
average_feature_importance_scores = feature_importance_scores / k

# Create a dictionary mapping feature names to their average importance scores
feature_importance_dict = dict(zip(numeric_features, average_feature_importance_scores))

# Sort features by their average importance scores in descending order
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Print the ranking of key variables
print("Ranking of key variables:")
for rank, (numeric_feature, importance_score) in enumerate(sorted_features, start=1):
    print(f"{rank}. {numeric_feature}: {importance_score}")


# In[226]:


import matplotlib.pyplot as plt

# Extracting sorted feature names and importance scores
sorted_numeric_features = [feature[0] for feature in sorted_features]
sorted_importance_scores = [feature[1] for feature in sorted_features]

# Plotting feature importance scores
plt.figure(figsize=(10, 6))
bars = plt.barh(sorted_numeric_features, sorted_importance_scores, color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Numeric Feature')
plt.title('Feature Importance Scores (Elastic-Net)')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important features at the top

# Add importance scores on the right-hand side of each bar
for bar, score in zip(bars, sorted_importance_scores):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{score:.2f}', 
             va='center', ha='left', fontsize=0.75*plt.rcParams['font.size'])

plt.show()


# ## 8.4 Decision Tree -  Classification and Regression Trees (CART)
# Besides using linear regression, we also try another method here to evaluate the importance of difference variables, which is one of the most popular decision tree algorithms -- CART. Among different decision tree algorithms, we chose CART becasue it can deal with both continuous and categorical features. And it provides a measure of variable importance, which indicates how much each feature contributes to the model's predictions.

# In[227]:


from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


# In[228]:


numeric_features = ['administrative_power', 'maximum_power_kW', 'urban_fuel_consumption',
                    'extra_urban_fuel_consumption', 'mixed_fuel_consumption', 'min_euro_empty_weight',
                    'max_euro_empty_weight']
X_train_numeric_features = X_train[numeric_features]
y_train = y_train


# In[229]:


# Initialize Decision Tree model
tree_reg = DecisionTreeRegressor(random_state=42)

# Train the Decision Tree model
tree_reg.fit(X_train_numeric_features, y_train)

# Extract feature importances
importances = tree_reg.feature_importances_

# Create a dictionary mapping feature names to their importance scores
feature_importance_dict = dict(zip(numeric_features, importances))

# Sort features by their importance scores in descending order
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Extract sorted feature names and importance scores
sorted_feature_names = [feature[0] for feature in sorted_features]
sorted_importance_scores = [feature[1] for feature in sorted_features]

# Plot feature importances
plt.figure(figsize=(10, 6))
bars = plt.barh(sorted_feature_names, sorted_importance_scores, color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Numeric Feature')
plt.title('Feature Importance Scores (CART)')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important features at the top

# Add importance scores on the right-hand side of each bar
for bar, score in zip(bars, sorted_importance_scores):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{score:.2f}', 
             va='center', ha='left', fontsize=0.75*plt.rcParams['font.size'])

plt.show()


# ## 8.5 Random Forest
# (Based on the result in the CART model, we trained the random forest)
# 
# But there is drawback in decision tree algorithms, which is high variance. They does not generalize
# the pattern well, and will lead to overfitting. To address this problem, we use ensemble methods which combine several predictive models to produce optimal predictive model.
# 
# Here we use Random Forest, because it can add more randomness, further reduce the variance, and thus provide more robust estimates of variable importance compared to a single decision tree like CART.
# 
# And we can evaluate the importance of variables in random forest by looking at how frequently the variables are used by trees. The more they are used at the 'top' of the trees, the more important the variable is.

# In[230]:


from sklearn.ensemble import RandomForestRegressor


# In[231]:


# Extract the selected variables from the sorted feature list (e.g., top 5)
selected_variables = [feature[0] for feature in sorted_features[:7]]

X_train_selected = X_train_numeric_features[selected_variables]

# Initialize Random Forest model
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the Random Forest model using the selected variables
forest_reg.fit(X_train_selected, y_train)

# Extract feature importances
importances = forest_reg.feature_importances_

# Plot feature importances
plt.figure(figsize=(10, 6))
bars = plt.barh(selected_variables, importances, color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Numeric Feature')
plt.title('Feature Importance Scores (Random Forest)')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important features at the top

# Add importance scores on the right-hand side of each bar
for bar, score in zip(bars, importances):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{score:.2f}', 
             va='center', ha='left', fontsize=0.8*plt.rcParams['font.size'])

plt.show()


# # 9. Cross-Check Result
# 
# Ranking of importance of key variables:
# 1. mixed_fuel_consumption
# 2. urban_fuel_consumption
# 3. extra_urban_fuel_consumption
# 4. max_euro_empty_weight
# 5. administrative_power
# 6. maximum_power_kW
# 7. min_euro_empty_weight

# In[232]:


import seaborn as sns


# In[233]:


# Concatenate independent and dependent variables
features = ['administrative_power', 'maximum_power_kW', 'urban_fuel_consumption',
            'extra_urban_fuel_consumption', 'mixed_fuel_consumption', 'min_euro_empty_weight',
            'max_euro_empty_weight']

X_train_features = X_train[features]

data = X_train_features.copy()
data['co2_gkm'] = y_train

# Calculate correlation coefficients
correlation_matrix = data.corr()

# Extract correlations with the dependent variable
correlation_with_target = correlation_matrix['co2_gkm'].drop('co2_gkm')

# Visualize the correlation coefficients
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_with_target.to_frame(), annot=True, cmap='coolwarm', cbar=False)
plt.title('Correlation Coefficients with co2_gkm')
plt.ylabel('Numeric Feature')
plt.show()


# In[234]:


# Define the features for correlation analysis
features = ['administrative_power', 'maximum_power_kW', 'urban_fuel_consumption',
            'extra_urban_fuel_consumption', 'mixed_fuel_consumption', 'min_euro_empty_weight',
            'max_euro_empty_weight', 'hybrid', 'co2_gkm']

# Create a correlation matrix
correlation_matrix = df[features].corr()

# Plot the correlation matrix using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Matrix', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()


# ## 10. Prediction Model

# In[235]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import ipywidgets as widgets
from ipywidgets import interact, FloatSlider, Layout, fixed


# In[236]:


numeric_features = ['administrative_power', 'maximum_power_kW', 'urban_fuel_consumption',
                    'extra_urban_fuel_consumption', 'mixed_fuel_consumption', 'min_euro_empty_weight',
                    'max_euro_empty_weight']
X_train_numeric_features = X_train[numeric_features]
y_train = y_train


# In[237]:


# Define the model
model = RandomForestRegressor()
model.fit(X_train_numeric_features, y_train)

# Define function for prediction and showing results
def predict_and_show(administrative_power, maximum_power_kW, urban_fuel_consumption,
                     extra_urban_fuel_consumption, mixed_fuel_consumption, min_euro_empty_weight,
                     max_euro_empty_weight):
    # Create a dataframe from input values
    data = pd.DataFrame([[administrative_power, maximum_power_kW, urban_fuel_consumption,
                          extra_urban_fuel_consumption, mixed_fuel_consumption, min_euro_empty_weight,
                          max_euro_empty_weight]],
                        columns=['administrative_power', 'maximum_power_kW', 'urban_fuel_consumption',
                                 'extra_urban_fuel_consumption', 'mixed_fuel_consumption', 'min_euro_empty_weight',
                                 'max_euro_empty_weight'])
    # Predict using the model
    prediction = model.predict(data)
    # Show the prediction
    print("Predicted CO2 emissions (g/km):", prediction[0])

# Create sliders for each variable with larger size and font
layout = Layout(width='60%')  # Adjust the width as needed
interact(predict_and_show,
         administrative_power=FloatSlider(min=0, max=20, step=0.1, value=11, description='Administrative Power:', 
                                           layout=layout, style={'description_width': 'initial'}),
         maximum_power_kW=FloatSlider(min=0, max=500, step=1, value=135, description='Maximum Power (kW):', 
                                       layout=layout, style={'description_width': 'initial'}),
         urban_fuel_consumption=FloatSlider(min=0, max=10, step=0.1, value=5.5, description='Urban Fuel Consumption:', 
                                             layout=layout, style={'description_width': 'initial'}),
         extra_urban_fuel_consumption=FloatSlider(min=0, max=10, step=0.1, value=2.5, 
                                                   description='Extra Urban Fuel Consumption:', 
                                                   layout=layout, style={'description_width': 'initial'}),
         mixed_fuel_consumption=FloatSlider(min=0, max=10, step=0.1, value=1.5, description='Mixed Fuel Consumption:', 
                                            layout=layout, style={'description_width': 'initial'}),
         min_euro_empty_weight=FloatSlider(min=0, max=5000, step=10, value=1774, description='Min Euro Empty Weight:', 
                                            layout=layout, style={'description_width': 'initial'}),
         max_euro_empty_weight=FloatSlider(min=0, max=5000, step=10, value=1826, description='Max Euro Empty Weight:', 
                                            layout=layout, style={'description_width': 'initial'}))


# In[238]:


df.describe()


# In[50]:


desktop_path = "/Users/CHIENWEI/Desktop/"


# In[51]:


filename = 'CO2_emissions_data_for_copulas.csv'
filepath = desktop_path + filename


# In[52]:


df.to_csv(filepath, index=False)


# In[ ]:




