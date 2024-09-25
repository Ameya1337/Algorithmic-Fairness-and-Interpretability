import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_excel("../Data/dataproject2024.xlsx")

# Show the first few rows of the DataFrame
print("First 5 rows of the dataset:")
print(data.head())

# Basic statistical summary of the numerical columns
print("\nStatistical summary:")
print(data.describe())

# Checking for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Correlation matrix to identify relationships between variables
print("\nCorrelation matrix:")
correlation_matrix = data.corr()
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Distribution of the 'PD' (Probability of Default)
plt.figure(figsize=(8, 5))
sns.histplot(data['PD'], kde=True)
plt.title('Distribution of Probability of Default (PD)')
plt.show()

# Boxplot of 'PD' by 'Default (y)' to understand its impact
plt.figure(figsize=(8, 5))
sns.boxplot(x='Default (y)', y='PD', data=data)
plt.title('Boxplot of PD by Default (y)')
plt.show()

# Countplot of Default (y) to understand how many defaults occurred
plt.figure(figsize=(6, 4))
sns.countplot(x='Default (y)', data=data)
plt.title('Count of Default (y)')
plt.show()

# Pairplot to visualize relationships between key variables
sns.pairplot(data[['Age', 'Job tenure', 'Car price', 'Funding amount', 'PD', 'Default (y)']])
plt.show()