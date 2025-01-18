import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from mlxtend.plotting import plot_decision_regions

# Configuration options
num_samples_total = 10000
cluster_centers = [(5,5), (3,3), (1,5)]
num_classes = len(cluster_centers)

# Generate data
X, y = make_blobs(n_samples = num_samples_total, centers = cluster_centers, n_features = num_classes, center_box=(0, 1), cluster_std = 0.30)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# np.save('./clusters.npy', X)
X = np.load('./clusters.npy')

# Create the SVM
svm = LinearSVC(random_state=42)

# Make it an OvO classifier
ovo_classifier = OneVsOneClassifier(svm)

# Fit the data to the OvO classifier
ovo_classifier = ovo_classifier.fit(X_train, y_train)

# Evaluate by means of a confusion matrix
matrix = ConfusionMatrixDisplay(ovo_classifier, X_test, y_test,cmap=plt.cm.Blues,normalize='true')
plt.title('Confusion matrix for OvO classifier')
plt.show()

# Plot decision boundary
plot_decision_regions(X_test, y_test, clf=ovo_classifier, legend=2)
plt.show()