# Prostate-Cancer-Prediction
Model Development:
I used the K-nearest neighbors (KNN) algorithm, a supervised learning technique that classifies data points based on their proximity to other points in the feature space. In this case, I used 100 computed tomography (CT) scans, extracting 8 numerical attributes (such as pixel intensity, texture features, etc.) to serve as input features for the model.

Cross Validation:
To enhance the generalizability of the model, I employed Random Cross Validation, a technique where the dataset is split into multiple subsets, and the model is trained and validated on different partitions. This ensures that the model is not overfitting to a specific dataset partition.
I also optimized hyperparameters, such as the value of k in KNN, to improve the performance of the model, leading to a final accuracy rate of 98%.

Model Evaluation:
The performance of the model was evaluated using metrics such as accuracy and confusion matrices to assess its ability to correctly predict benign and malignant cases.
The high accuracy of the model demonstrates that the feature engineering and the choice of KNN, which considers the Euclidean distance between points, was effective for this classification task.
