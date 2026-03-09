import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.impute import SimpleImputer

from sklearn.datasets import fetch_california_housing

#Load Dataset
housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Price"] = housing.target

#1.Data overview
print("\nDataset Shape:", df.shape)
print("\nFirst 5 rows of Dataset:")
print(df.head(5))

#2.Dataset Information
print("\nDataset Info:")
print(df.info())

#3.Summary Statistics
print("\nSummary statistics:")
print(df.describe())

#4. Missing Values
print("\nMissing values:")
print(df.isnull().sum())

#5.Correlation Matrix
corr_matrix = df.corr()
print("\nCorrelation matrix:")
print(corr_matrix)

#6.Plot Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm")
plt.title("Feature Correlation Matrix")
# plt.show()

#7.Distribution of features
df.hist(bins=30, figsize=(12,8))    #create histograms for each feature
plt.suptitle("Feature distributions")
plt.show()

#8.Scatter plot
plt.figure(figsize=(10,8))
plt.scatter(df["MedInc"],df["Price"], alpha = 0.3)
plt.xlabel("Median Income")
plt.ylabel("Price")
plt.title("Price vs Median Income")
plt.show()

#9.Boxplot for outlier detection
plt.figure(figsize=(12,6))
sns.boxplot(data=df)
plt.title("Box Plot for outlier detection")
plt.show()


#10.Detect outliers using IQR
for col in df.columns:

    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)

    IQR = Q3-Q1             #represemts the middle 50% of data #IQR = Inter Quartile Range

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]   #filters rows where the feature value is outside the allowed range

    print(f"{col}Outliers:", len(outliers))

#11. Geographic visualization

plt.figure(figsize=(10,8))
plt.scatter(df["Longitude"], df["Latitude"], alpha = 0.3)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("California Housing Price Map")
plt.show()