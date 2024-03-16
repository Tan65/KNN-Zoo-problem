# KNN-Zoo-problem
Implement a KNN model to classify the animals in to categorie
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Step 1: Read the CSV file into a DataFrame
data = pd.read_csv('Zoo.csv')

# Drop the 'animal name' column as it contains non-numeric values
data = data.drop('animal name', axis=1)

# Step 2: Split the data into features and target
X = data.drop('type', axis=1)
y = data['type']

# Step 3: Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Step 4: Split the normalized data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Step 5: Create a KNN classifier
knn = KNeighborsClassifier()

# Step 6: Fit the model to the training data
knn.fit(X_train, y_train)

# Step 7: Predict the target values for the test data
y_pred = knn.predict(X_test)

# Step 8: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 9: Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 10: Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 11: Exploratory Data Analysis

# Univariate Analysis
num_cols = X.columns
num_plots = len(num_cols)
num_rows = (num_plots // 3) + (1 if num_plots % 3 != 0 else 0)  # Calculate number of rows needed
plt.figure(figsize=(20, num_rows * 5))
for i, col in enumerate(num_cols):
    plt.subplot(num_rows, min(num_plots, 3), i+1)
    sns.histplot(X[col], kde=True)
    plt.title(f'Histogram of {col}')
plt.tight_layout()
plt.show()

# Bivariate Analysis (Scatter Plots)
plt.figure(figsize=(15, 10))
for i, col in enumerate(X.columns):
    plt.subplot(num_rows, min(num_plots, 3), i+1)
    sns.scatterplot(x=col, y='type', data=data)
    plt.title(f'Scatter plot of {col} vs type')
plt.tight_layout()
plt.show()

# Multivariate Analysis (Correlation Heatmap)
plt.figure(figsize=(10, 8))
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
