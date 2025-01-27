# Import needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv("Iris.csv")

# Drop the 'Id' column as it's not useful for prediction
data = data.drop(columns=['Id'])

# Encode the target variable
label_encoder = LabelEncoder()
data['Species'] = label_encoder.fit_transform(data['Species'])

# Split the data into features and target variable
X = data.drop(columns=['Species'])
y = data['Species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression Model
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)

# Evaluate Logistic Regression
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
print("Logistic Regression Performance:")
print("Accuracy:", logistic_accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, logistic_predictions))
print("Classification Report:\n", classification_report(y_test, logistic_predictions))

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
kmeans_labels = kmeans.labels_

# Evaluate Clustering
silhouette_avg = silhouette_score(X, kmeans_labels)
print("K-Means Clustering Performance:")
print(f"Silhouette Score: {silhouette_avg:.2f}")

# Map K-Means clusters to original species for comparison
def map_clusters_to_species(cluster_labels, true_labels):
    mapping = {}
    for cluster in np.unique(cluster_labels):
        true_labels_in_cluster = true_labels[cluster_labels == cluster]
        most_common_label = np.bincount(true_labels_in_cluster).argmax()
        mapping[cluster] = most_common_label
    return mapping

cluster_to_species = map_clusters_to_species(kmeans_labels, y.values)
data['Cluster'] = kmeans_labels
mapped_clusters = data['Cluster'].map(cluster_to_species)
kmeans_accuracy = accuracy_score(y, mapped_clusters)

print(f"Clustering Accuracy (mapped to species): {kmeans_accuracy:.2f}")

# Displaying the accuracies of both algorithms
print(f"\nSummary of Model Accuracies:")
print(f"Logistic Regression Accuracy: {logistic_accuracy:.2f}")
print(f"K-Means Clustering Accuracy (mapped): {kmeans_accuracy:.2f}")

# Binarize the output labels for multi-class ROC
y_test_binarized = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_binarized.shape[1]

# Logistic Regression - ROC Curve
logistic_prob = logistic_model.predict_proba(X_test)
fpr_lr = {}
tpr_lr = {}
roc_auc_lr = {}

for i in range(n_classes):
    fpr_lr[i], tpr_lr[i], _ = roc_curve(y_test_binarized[:, i], logistic_prob[:, i])
    roc_auc_lr[i] = auc(fpr_lr[i], tpr_lr[i])

# Plotting the ROC Curve for Logistic Regression
plt.figure(figsize=(12, 8))
colors = ['blue', 'green', 'red']
for i, color in enumerate(colors):
    plt.plot(fpr_lr[i], tpr_lr[i], color=color, lw=2, label=f"Logistic Regression (Class {i}, AUC = {roc_auc_lr[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line
plt.title("ROC Curve for Logistic Regression", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
