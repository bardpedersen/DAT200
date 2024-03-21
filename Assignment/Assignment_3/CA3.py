# %% [markdown]
# # CA3

# %% [markdown]
# ### Imports

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# %% [markdown]
# ### Reading data

# %%
df = pd.read_csv("assets/train.csv", index_col=0) # First column as row index

# %% [markdown]
# ### Data exploration and visualisation

# %%
# Visualize the data
sns.pairplot(df, hue="Edible")
plt.tight_layout()
plt.show()

# Print the mean, median, and standard deviation of each column
print(df.describe())

# %% [markdown]
# From the plots we can see that the data is not linearly separable, meaning that we will have a better model if we use a non-linear model. From the description of the data we can see that some columns are missing data, wich we will have to deal with. We can also see that the data is not normalized, so we will have to normalize it. We also can see outliers when the max value is much higher than the 75% percentile. We can also se that Circumference (mm), Length (mm) and Weight (mg) are highly correlated, so we can drop two of them. 

# %% [markdown]
# ### Data cleaning

# %%
# Replace missing values with the mean of the column
columns = df.columns
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(df.values)
df = imputer.transform(df.values)
df = pd.DataFrame(df, columns=columns)

"""
Removing the outlisers and correlated columns are commented out because they decrease the accuracy of the model.
"""
# Replace outliers with the median of the column
#for i in df.columns:
#   z_scores = (df[i] - np.mean(df[i])) / np.std(df[i])
#   df.loc[np.abs(z_scores) > 3, i] = df[i].mean()

# Remove correlated columns
#df.drop(columns=["Weight (mg)"], inplace=True)
#df.drop(columns=["Circumference (mm)"], inplace=True)
print(df.describe()) # To check if the outliers and missing values are replaced


# %% [markdown]
# Here we can see that all the columns have the same number of items, meaning that we replaced the missing values. Would also se that max value would be lower if outliers were removed.

# %% [markdown]
# ### Data preprocessing and visualisation

# %%
# Split training into training and testing data
y = df["Edible"]
X = df.drop(columns="Edible")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)

# Scale and standardize the data for better performance on KNN
sc = StandardScaler()
sc.fit(X_train)

X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)

# Plot scaled data as violin plot
X_train__copy = pd.DataFrame(X_train_sc, columns=X.columns)
sns.violinplot(data=X_train__copy)

# %% [markdown]
# Here we can see that the data is standardized and that some features have outliers.

# %% [markdown]
# ### Modelling

# %%
# Test different number of estimators to find the best model
for i in range(1, 50):
    rf = RandomForestClassifier(n_estimators=i, random_state=0)
    rf.fit(X_train_sc, y_train)
    y_pred = rf.predict(X_test_sc)
    print("rf", accuracy_score(y_test, y_pred), i)

# Testing KNN as well
for i in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_sc, y_train)
    y_pred = knn.predict(X_test_sc)
    print("knn", accuracy_score(y_test, y_pred), i)

# %% [markdown]
# We kan see that Random Forest performs the best, so will tune the hyperparameters of this model.

# %%
best = 0
for i in range(1, 50):
    rf = RandomForestClassifier(n_estimators=i, random_state=0)
    rf.fit(X_train_sc, y_train)
    y_pred = rf.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    if acc > best:
        print(acc, i)
        best = acc

best = 0
for j in range(0, 50):
    rf = RandomForestClassifier(n_estimators=37, random_state=j)
    rf.fit(X_train_sc, y_train)
    y_pred = rf.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    if acc > best:
        print(acc, j)
        best = acc

# %% [markdown]
# ### Final evaluation

# %%
df = pd.read_csv("assets/train.csv", index_col=0) 

# Transform and mangage the data as before
columns = df.columns
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(df.values)
df = imputer.transform(df.values)
df = pd.DataFrame(df, columns=columns)


# Testing on the whole dataset
df_y = df["Edible"]
df_x = df.drop(columns="Edible")

# Scale and standardize the data for better performance
sc = StandardScaler()
sc.fit(df_x)
df_x_sc = sc.transform(df_x)

rf = RandomForestClassifier(n_estimators=18, random_state=0) # Choose random state as 0 since the other values created overfitting
rf.fit(df_x_sc, df_y)

df_train_ped = rf.predict(df_x_sc)
print(accuracy_score(df_train_ped, df_y)) # Test accuracy for training data

# %% [markdown]
# ### Kaggle submission

# %%
df = pd.read_csv("assets/test.csv", index_col=0)

# Transform and mangage the data as before
columns = df.columns
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(df.values)
df = imputer.transform(df.values)
df = pd.DataFrame(df, columns=columns)

# Scale and standardize the data the same way as the training data
df_test_sc = sc.transform(df)
df_test_pred = rf.predict(df_test_sc)

# Save the results to a csv file
df_results = pd.DataFrame(data=df_test_pred.astype(int), columns=["Edible"])
df_results.index.names = ["index"]
df_results.to_csv('assets/results.csv')


