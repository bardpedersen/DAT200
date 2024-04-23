# %% [markdown]
# # CA5

# %% [markdown]
# ### Imports

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression

# %% [markdown]
# ### Reading data

# %%
df = pd.read_csv('assets/train.csv')

# %% [markdown]
# ### Data exploration and visualisation

# %%
# Check some basic information
print(df.describe())

# check for missing values
print(df.isnull().sum())

# Plot pairplot
sns.pairplot(df, hue='Scoville Heat Units (SHU)')
plt.show()

# Plot bins with target varible
sns.histplot(df['Scoville Heat Units (SHU)'], bins=100)
plt.show()

# %% [markdown]
# From the data exploration we can see that 648 rows are missing in the `Average Temperature During Storage (celcius)` column. Since this is aproximately 65% of the data, i decided to drop the column. From the pairplot and the description of data we can see that most data is uniformly distributed, witout to many outliers. I am still going to remove them. From the pairplot we can also see that the data is not linearly separable and that the data is not very correlated.  
# We can also see that most of the target values are at 0. This indecates that the best model might be to use one model for the 0 values and another model for the non-zero values. 

# %% [markdown]
# ### Turn categorical variables into numerical 

# %%
# Create two lists, one for numeric columns and one for categorical columns
numeric_columns = np.array(df.select_dtypes(include=['float64', 'int64']).columns)

category_columns = np.array(df.select_dtypes(include=['object']).columns)

print(numeric_columns)
print(category_columns)

# %% [markdown]
# ### Data cleaning

# %% [markdown]
# #### Remove missing values and outliers
# 
# 

# %%
# Remove column 'Average Temperature During Storage (celcius)' because it has a lot of missing values
df = df.drop(columns=['Average Temperature During Storage (celcius)'])
category_columns = np.delete(category_columns, np.where(category_columns == 'Average Temperature During Storage (celcius)'))

print(category_columns) # Check if column was removed

# %% [markdown]
# ### Data preprocessing and visualisation

# %% [markdown]
# #### Split data

# %%
y = df['Scoville Heat Units (SHU)']
df = df.drop(columns=['Scoville Heat Units (SHU)'])
numeric_columns = np.delete(numeric_columns, np.where(numeric_columns == 'Scoville Heat Units (SHU)'))

# Replace outliers with the mean
for i in numeric_columns:
    z_score = (df[i] - np.mean(df[i])) / np.std(df[i])    
    df.loc[np.abs(z_score) > 4, i] = df[i].mean()

X = df
# Need to remove the target variable from the numeric columns, shall not be scaled
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %% [markdown]
# ### Modelling

# %%
# Create a transformer for the numeric columns
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')), # fill missing values with the mean of the column
    ('scaler', StandardScaler())
])

# Create a transformer for the category columns
category_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), # fill missing values with the most frequent value of the column
    ('onehot', OneHotEncoder())
])

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', category_transformer, category_columns)
    ]
)

# %%
# Create a transformer for the numeric columns, but with pca for random forest
numeric_transformer_pca = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('pca', PCA())
])

# Create a preprocessor
preprocessor_pca = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_pca, numeric_columns),
        ('cat', category_transformer, category_columns)
    ]
)

# %% [markdown]
# #### PLS

# %%
# Create a pipeline for the PLS model
pls = Pipeline([
    ('preprocessor', preprocessor),
    ('pls', PLSRegression())
])

# Create a parameter grid for the PLS model
param_grid = {
    'pls__n_components': [1, 2, 3, 4, 5, 6 ,7, 8, 10],
    'pls__scale': [True, False],
    'pls__max_iter': [100, 500, 1000, 1500],
}

# Create a grid search for the PLS model
pls_grid_search = GridSearchCV(estimator = pls, 
                               param_grid = param_grid,
                               cv = 10, 
                               scoring = 'neg_mean_absolute_error', # Use mean absolute error beacuse we want to be as close to the target value as possible
                               n_jobs = -1,
                               verbose = 1)

pls_model = pls_grid_search.fit(X_train, y_train)

print(pls_model.best_score_)
print(pls_model.best_params_)

# %% [markdown]
# #### Random Forest and PLS

# %%
# Create a binary version for train data
y_train_binary = y_train.apply(lambda x: 0 if x == 0 else 1)

# %%
pipeline_clf = Pipeline([
    ('preprocessor', preprocessor_pca),
    ('rfc', RandomForestClassifier(random_state=42))
])

param_grid = {
    'rfc__n_estimators': [50, 100, 200, 400],
    'rfc__criterion': ['gini', 'entropy', 'log_loss'],
    'rfc__max_depth': [5, 10, 20],
    'rfc__min_samples_split': [2, 3, 5],
    'rfc__min_samples_leaf': [1, 2, 5],
    'preprocessor__num__pca__n_components': [0.75, 0.8, 0.85]
}

grid_search_clf = GridSearchCV(estimator = pipeline_clf,
                                 param_grid = param_grid,
                                 cv = 5,
                                 scoring = 'accuracy', # use accuracy, because we want as many correct predictions as possible
                                 n_jobs = -1,
                                 verbose = 1)

grid_search_clf.fit(X_train, y_train_binary)
print(grid_search_clf.best_score_)
print(grid_search_clf.best_params_)


# %%
# For those samples that are predicted as spicy peppers, train a regression model
X_train_spicy = X_train[y_train_binary == 1]
y_train_spicy = y_train[y_train_binary == 1]

pls = Pipeline([
    ('preprocessor', preprocessor),
    ('pls', PLSRegression())
])

param_grid = {
    'pls__n_components': [1, 2, 3, 4, 5, 6 ,7, 8, 10],
    'pls__scale': [True, False],
    'pls__max_iter': [100, 500, 1000, 1500],
}

pls_grid_search = GridSearchCV(estimator = pls, 
                               param_grid = param_grid,
                               cv = 10, 
                               scoring = 'neg_mean_absolute_error',
                               n_jobs = -1,
                               verbose = 1)

pls_grid_search.fit(X_train_spicy, y_train_spicy)
print(pls_grid_search.best_score_)
print(pls_grid_search.best_params_)


# %%
# Predict on the test data using the trained classification model
y_binary_pred = grid_search_clf.best_estimator_.predict(X_test)
# Predict on the test data using the trained regression model
y_pred = pls_grid_search.best_estimator_.predict(X_test)

# merge the two predictions into one, with y_binary_pred as the mask and where that is 1, use y_pred
y_final_pred = y_pred.copy()
y_final_pred[y_binary_pred == 0] = y_binary_pred[y_binary_pred == 0]

# Calculate the mean absolute error
mae = mean_absolute_error(y_test, y_final_pred)
print(mae)


# %% [markdown]
# ### Final evaluation

# %%
df = pd.read_csv('assets/train.csv')

# Remove column 'Average Temperature During Storage (celcius)' because it has a lot of missing values
df = df.drop(columns=['Average Temperature During Storage (celcius)'])

y = df['Scoville Heat Units (SHU)']

df = df.drop(columns=['Scoville Heat Units (SHU)'])

for i in numeric_columns:
    z_score = (df[i] - np.mean(df[i])) / np.std(df[i])    
    df.loc[np.abs(z_score) > 4, i] = df[i].mean()

X = df

# fit the models on the entire dataset
y_binary = y.apply(lambda x: 0 if x == 0 else 1)
grid_search_clf.best_estimator_.fit(X, y_binary)
pls_grid_search.best_estimator_.fit(X[y_binary == 1], y[y_binary == 1])

# check the mean squared error on the entire dataset
y_binary_pred = grid_search_clf.best_estimator_.predict(X)
y_pred = pls_grid_search.best_estimator_.predict(X)

y_final_pred = y_pred.copy()
y_final_pred[y_binary_pred == 0] = y_binary_pred[y_binary_pred == 0]

mae = mean_absolute_error(y, y_final_pred)
print(mae)


# %% [markdown]
# ### Kaggle submission

# %%
# Load the new dataset
test_data = pd.read_csv('assets/test.csv')

# Preprocess the test data in the same way as the training data
test_data = test_data.drop(columns=['Average Temperature During Storage (celcius)'])

for i in numeric_columns:
    z_score = (test_data[i] - np.mean(test_data[i])) / np.std(test_data[i])    
    test_data.loc[np.abs(z_score) > 4, i] = test_data[i].mean()


# Predict whether the peppers are spicy using the trained classification model
y_binary = grid_search_clf.best_estimator_.predict(test_data)

# For those samples that are predicted as spicy peppers, predict the SHU using the trained regression model
y_spicy = pls_grid_search.best_estimator_.predict(test_data)

# Combine the results of the classification and regression models into a single prediction vector
y_new = y_spicy.copy()
y_new[y_binary == 0] = y_binary[y_binary == 0]

# Save the results to a CSV file
df = pd.DataFrame(y_new, columns=['Scoville Heat Units (SHU)'])
df.to_csv('assets/results.csv', index=True, index_label='index')


