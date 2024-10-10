# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('/mnt/data/Prostate_Cancer.csv')

# Preprocess the dataset
# Convert 'diagnosis_result' to binary (M = 1, B = 0)
data['diagnosis_result'] = data['diagnosis_result'].map({'M': 1, 'B': 0})

# Separate features (X) and labels (y)
X = data[['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'symmetry', 'fractal_dimension']]
y = data['diagnosis_result']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialize the KNN classifier with Euclidean distance metric
knn = KNeighborsClassifier(metric='euclidean')

# Define the range of 'k' values to search over
param_grid = {'n_neighbors': np.arange(1, 21)}

# Set up cross-validation scheme using Stratified K-Folds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform grid search with cross-validation to find the optimal 'k'
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=cv, scoring='accuracy')

# Fit the model on the training data
grid_search.fit(X_train, y_train)

# Retrieve the best hyperparameters and estimator
best_k = grid_search.best_params_['n_neighbors']
best_knn = grid_search.best_estimator_
best_score = grid_search.best_score_

# Output the optimal 'k' and corresponding cross-validated accuracy
print(f"Optimal number of neighbors (k): {best_k}")
print(f"Best cross-validated accuracy: {best_score:.2%}")

# Evaluate the optimized model on the test set
y_pred = best_knn.predict(X_test)

# Calculate accuracy on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy: {test_accuracy:.2%}")

# Generate the confusion matrix to assess classification performance
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Provide a detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
