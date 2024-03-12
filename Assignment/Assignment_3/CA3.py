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

print(df.describe())

# %% [markdown]
# From the plots we can see that the data is not linearly separable, meaning that we will have a better model if we use a non-linear model.

# %% [markdown]
# ### Data cleaning

# %%
# Replace missing values with the mean of the column
columns = df.columns
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(df.values)
df = imputer.transform(df.values)
df = pd.DataFrame(df, columns=columns)

# Remove outliers
for i in df.columns:
    z_scores = (df[i] - np.mean(df[i])) / np.std(df[i])
    df_clean = df[np.abs(z_scores) < 1]

# %% [markdown]
# ### Data preprocessing and visualisation

# %%
# Split training into training and testing data
y = df["Edible"]
X = df.drop(columns="Edible")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)

# Scale and standardize the data for better performance
sc = StandardScaler()
sc.fit(X_train)

X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)

# %% [markdown]
# ### Modelling

# %%
# Test different number of estimators to find the best model
for i in range(1, 50):
    rf = RandomForestClassifier(n_estimators=i, random_state=0) #37, 22
    rf.fit(X_train_sc, y_train)
    y_pred = rf.predict(X_test_sc)
    print("rf", accuracy_score(y_test, y_pred), i)

# %% [markdown]
# Since multiple estimators give the same accuracy, we will use the one that has the least complexity, to make sure that the model is not overfitting.

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

for i in df.columns:
    z_scores = (df[i] - np.mean(df[i])) / np.std(df[i])
    df_clean = df[np.abs(z_scores) < 1]

# Testing on the whole dataset
df_y = df["Edible"]
df_x = df.drop(columns="Edible")

# Scale and standardize the data for better performance
sc = StandardScaler()
sc.fit(df_x)
df_x_sc = sc.transform(df_x)

rf = RandomForestClassifier(n_estimators=18, random_state=0) # n_estimators found from previous test
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

for i in df.columns:
    z_scores = (df[i] - np.mean(df[i])) / np.std(df[i])
    df_clean = df[np.abs(z_scores) < 1]

# Scale and standardize the data the same way as the training data
df_test_sc = sc.transform(df)
df_test_pred = rf.predict(df_test_sc)

# Save the results to a csv file
df_results = pd.DataFrame(data=df_test_pred.astype(int), columns=["Edible"])
df_results.index.names = ["index"]
df_results.to_csv('assets/results.csv')


