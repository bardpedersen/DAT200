# %% [markdown]
# # CA4

# %% [markdown]
# ### Imports

# %%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# %% [markdown]
# ### Reading data

# %%
df = pd.read_csv('assets/train.csv', index_col=1) # Use 'index' column as index
df = df.drop(df.columns[0], axis=1) # Drop 'Unnamed: 0' column

# %% [markdown]
# ### Data exploration and visualisation

# %%
# To print out all the columns
pd.set_option('display.max_columns', None)
              
# Print the mean, median, and standard deviation of each column
print(df.describe())


# Plot the pairplot of the data
sns.pairplot(df, hue='Diagnosis')
plt.show()

# %% [markdown]
# From the plots we can see that the data is not linearly separable, meaning that we will have a better model if we use a non-linear model. From the description of the data we can see that none columns are missing data. We can also see that the data is not normalized, so we will have to normalize it. We also can see outliers when the max value is much higher than the 75% percentile. We can also se that some columns have a correlation, but these will most likely be removed with the PCA.

# %% [markdown]
# ### Data cleaning

# %% [markdown]
# #### Turn categorical variables into numerical

# %%
# Turn the Categorical variables into number variables
label_to_number = LabelEncoder()
df['Alcohol_Use (yes/no)'] = label_to_number.fit_transform(df['Alcohol_Use (yes/no)'])
df['Diabetes (yes/no)'] = label_to_number.fit_transform(df['Diabetes (yes/no)'])
df['Obesity (yes/no)'] = label_to_number.fit_transform(df['Obesity (yes/no)'])
df['Gender'] = label_to_number.fit_transform(df['Gender'])

# Take this last to be able to transform back.
df['Diagnosis'] = label_to_number.fit_transform(df['Diagnosis']) # ".fit_transform" is short for ".fit" and then ".transform"

# Split data into numerical and categorical columns for scaling.
numerical_column = [
    'AFP (ng/mL)',
    'ALP (U/L)',
    'ALT (U/L)',
    'AST (U/L)',
    'Age',
    'Albumin (g/dL)',
    'Bilirubin (mg/dL)',
    'CRP (mg/L)',
    'Fibroscan (kPa)',
    'GGT (U/L)',
    'Hemoglobin (g/dL)',
    'IL-6 (pg/mL)',
    'PT/INR',
    'Platelets (10^9/L)',
    'RBC (10^12/L)',
    'Serum_Ammonia (μmol/L)',
    'Serum_Copper (μg/dL)',
    'Serum_Creatinine (mg/dL)',
    'Serum_Iron (μg/dL)',
    'Serum_Lactate (mmol/L)',
    'Serum_Urea (mg/dL)',
    'Serum_Zinc (μg/dL)',
    'TIBC (μg/dL)',
    'Transferrin_Saturation (%)',
    'WBC (10^9/L)',
    'pH'
]

categorical_column = ['Alcohol_Use (yes/no)',
                      'Diabetes (yes/no)',
                      'Obesity (yes/no)',
                      'Gender']

print(df.head()) # To see that all the categorical columns are now numerical


# %% [markdown]
# #### Remove missing values and outliers
# 
# 

# %%
# Replace the outliers with the mean of the column
for i in numerical_column:
    z_score = (df[i] - np.mean(df[i])) / np.std(df[i])    
    df.loc[np.abs(z_score) > 4, i] = df[i].mean()


# Check if the outliers are replaced
print(df.describe())

# %% [markdown]
# ### Data preprocessing and visualisation

# %% [markdown]
# #### Split data

# %%
y = df['Diagnosis']
X = df.drop(columns=['Diagnosis'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,
    stratify=y, random_state=43)

# %% [markdown]
# ### Modelling

# %%
# Sett up pipleine
pipeline_svc = Pipeline([('preprocessor', ColumnTransformer(
                            transformers=[('num', StandardScaler(), numerical_column)], # Only scale the numerical values
                            remainder='passthrough')), # Skip the columns that are original categorical
                        ('pca', PCA()), # PCA to reduce the number of features
                        ('svc', SVC(max_iter=100000))]) # Model we train with max_iterations so it dont take to long. # Model using kernal

C_range     = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] # For regularization parameter C.
gamma_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]         # For scaling parameter gamma in rbf-kernel.
random_state = [1, 10, 50, 100, 200, 500, 700, 1000]               # For random state.
kernal_range = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] # For kernel type.

PCA_range = [0.90, 0.95, 0.99] # For PCA components.

# Create a paramgrid with all differnet varibles to combine
param_grid = [{'svc__C': C_range, 'svc__kernel': kernal_range, 'svc__gamma': gamma_range, 'svc__random_state': random_state, 
               'pca__n_components': PCA_range}] 

# Set up the grid search
gs_svc = GridSearchCV(estimator=pipeline_svc, # Use the pipeline we created
                  param_grid=param_grid,  # Use the different params we created
                  scoring='f1_macro',  # Sett F1 macro as score
                  cv=10, # Number of cross validaions
                  n_jobs=-1, # Use all avalible cores
                  verbose=1) # Print total number of combinations to itterate thru.

# Fit the model
gs_svc_res = gs_svc.fit(X_train, y_train) 

# Print results as best params and its corresponding score
print(gs_svc_res.best_score_)
print(gs_svc_res.best_params_)


# %%
pipeline_lr = Pipeline([('preprocessor', ColumnTransformer(
                            transformers=[('num', StandardScaler(), numerical_column)],
                            remainder='passthrough')), 
                        ('pca', PCA()),
                        ('lr', LogisticRegression(max_iter=10000))]) # Model using regularization


multi_class = ['auto']
C =  [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
random_state = [1, 10, 50, 100, 200, 500, 700, 1000]
PCA_range = [0.80, 0.85, 0.90, 0.95, 0.99]

# Need to split into multiple param_grids because of the different solvers dont work with all the penalty types.
param_grid = [{'lr__penalty': ['l2'],
                'lr__C': C,
                'lr__solver': ['lbfgs','newton-cg','newton-cholesky','sag'],
                'lr__multi_class': multi_class,
                'lr__random_state': random_state,
                'pca__n_components': PCA_range},
                {'lr__penalty': ['l1', 'l2'],
                'lr__C': C,
                'lr__solver': ['liblinear'],
                'lr__multi_class': multi_class,
                'lr__random_state': random_state,
                'pca__n_components': PCA_range},
                {'lr__penalty': ['elasticnet','l1', 'l2'],
                'lr__C': C,
                'lr__l1_ratio': [0.1, 0.5, 0.9],
                'lr__solver': ['saga'],
                'lr__multi_class': multi_class,
                'lr__random_state': random_state,
                'pca__n_components': PCA_range},]

gs_lr = GridSearchCV(estimator=pipeline_lr, 
                  param_grid=param_grid, 
                  scoring='f1_macro', 
                  cv=10,
                  n_jobs=-1,
                  verbose=1)

gs_lr_res = gs_lr.fit(X_train, y_train)
print(gs_lr_res.best_score_)
print(gs_lr_res.best_params_)


# %%
#split agian wiht 60/40
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4,
    stratify=y, random_state=42)

# Use best classifier
best_clf = gs_lr_res.best_estimator_
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

# Create a confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

# %% [markdown]
# ### Final evaluation

# %%
# Load datasett agin for now we want to train on all data
df = pd.read_csv('assets/train.csv')
df = df.drop(df.columns[0], axis=1)
df.set_index('index', inplace=True)

# Turn the Categorical variables into number variables
label_to_number = LabelEncoder()
df['Alcohol_Use (yes/no)'] = label_to_number.fit_transform(df['Alcohol_Use (yes/no)'])
df['Diabetes (yes/no)'] = label_to_number.fit_transform(df['Diabetes (yes/no)'])
df['Obesity (yes/no)'] = label_to_number.fit_transform(df['Obesity (yes/no)'])
df['Gender'] = label_to_number.fit_transform(df['Gender'])

df['Diagnosis'] = label_to_number.fit_transform(df['Diagnosis']) 

y = df['Diagnosis']
X = df.drop(columns=['Diagnosis'])


for i in numerical_column:
    z_score = (df[i] - np.mean(df[i])) / np.std(df[i])    
    df.loc[np.abs(z_score) > 4, i] = df[i].mean()

gs_svc_res.best_estimator_.fit(X, y)
print(f'Train accuracy: {gs_svc_res.best_estimator_.score(X, y)}')

# %% [markdown]
# ### Kaggle submission

# %%
df = pd.read_csv("assets/test.csv")
df = df.drop(df.columns[0], axis=1)
df.set_index('index', inplace=True)

gender_mapping = {'MALE': 0, 'FEMALE': 1}
df['Gender'] = df['Gender'].map(gender_mapping)

yes_no_mapping = {'no': 0, 'yes': 1}
df['Alcohol_Use (yes/no)'] = df['Alcohol_Use (yes/no)'].map(yes_no_mapping)
df['Diabetes (yes/no)'] = df['Diabetes (yes/no)'].map(yes_no_mapping)
df['Obesity (yes/no)'] = df['Obesity (yes/no)'].map(yes_no_mapping)

for i in numerical_column:
    z_score = (df[i] - np.mean(df[i])) / np.std(df[i])    
    df.loc[np.abs(z_score) > 4, i] = df[i].mean()

y = gs_svc_res.best_estimator_.predict(df)
y = label_to_number.inverse_transform(y) # Turn back into categorical values
df_results = pd.DataFrame(data=y, columns=["Diagnosis"])
df_results.index.names = ["index"]
df_results.to_csv('assets/results.csv') # Save results to csv

# %% [markdown]
# ### Binary version of the target variables

# %%
df = pd.read_csv('assets/train.csv', index_col=1)
df = df.drop(df.columns[0], axis=1)

# Turn the Categorical variables into number variables

label_to_number = LabelEncoder()
df['Alcohol_Use (yes/no)'] = label_to_number.fit_transform(df['Alcohol_Use (yes/no)'])
df['Diabetes (yes/no)'] = label_to_number.fit_transform(df['Diabetes (yes/no)'])
df['Obesity (yes/no)'] = label_to_number.fit_transform(df['Obesity (yes/no)'])
df['Gender'] = label_to_number.fit_transform(df['Gender'])

# print all values in diagnosis
print(df['Diagnosis'].unique())

Diagnosis_mapping = {'Healthy': 0, 
                     'Cirrhosis': 1, 'Drug-induced Liver Injury':1, 'Fatty Liver Disease': 1, 
                     'Hepatitis': 1, 'Autoimmune Liver Diseases': 1, 'Liver Cancer' : 1}
df['Diagnosis'] = df['Diagnosis'].map(Diagnosis_mapping)
print(df.head()) # Check if its only 0 and 1 in Diagnosis column.

# %%
# Split the data into X and y
y = df['Diagnosis']
X = df.drop(columns=['Diagnosis'])

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


pipeline_lr = Pipeline([('preprocessor', ColumnTransformer(
                            transformers=[('num', StandardScaler(), numerical_column)],
                            remainder='passthrough')), 
                        ('pca', PCA()),
                        ('lr', LogisticRegression(max_iter=10000))])


multi_class = ['auto']
C =  [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0] 
PCA_range = [0.80, 0.85, 0.90, 0.95]

param_grid = [{'lr__penalty': ['elasticnet','l1', 'l2'],
                'lr__C': C,
                'lr__solver': ['saga'],
                'lr__multi_class': multi_class,
                'lr__random_state': random_state,
                'pca__n_components': PCA_range}]

gs_lr = GridSearchCV(estimator=pipeline_lr, 
                  param_grid=param_grid, 
                  scoring='f1_macro', 
                  cv=10,
                  n_jobs=-1,
                  verbose=1)

# Print results
gs_lr_res = gs_lr.fit(X_train, y_train)
print(gs_lr_res.best_score_)
print(gs_lr_res.best_params_)

# %%
best_clf = gs_lr_res.best_estimator_
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

# %%
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()


# Set up pipeline with scale, decomposer and classifier
pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=0.90),
                        LogisticRegression(penalty='l2', 
                                           random_state=1, 
                                           C=0.1,
                                           solver='saga'))

# Cross-validation predictions
probas = cross_val_predict(pipe_lr, X_train, y_train, cv=5, method='predict_proba')

# False Positive and True Positive Rates
fpr, tpr, thresholds = roc_curve(y_train, probas[:, 1], pos_label=1)

# ROC AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
fig = plt.figure(figsize=(7, 5))

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')
plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


