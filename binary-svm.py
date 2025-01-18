from sklearn.datasets import make_blobs # it gives me the data
from sklearn.model_selection import train_test_split # function to help to split the data
import numpy as np # numbers processing
import matplotlib.pyplot as plt # plotting
from sklearn import svm
from sklearn.metrics import ConfusionMatrixDisplay
from mlxtend.plotting import plot_decision_regions

blobs_random_seed = 42
centers = [(0,0), (5,5)]
cluster_std = 1
frac_test_split = 0.33
num_features_for_samples = 2
num_samples_total = 1000

# Generate data
X, y = make_blobs(n_samples = num_samples_total, centers = centers, n_features = num_features_for_samples, cluster_std = cluster_std)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=frac_test_split, random_state=blobs_random_seed)

'''
np.save('./data.npy', (X_train, X_test, y_train, y_test))
X_train, X_test, y_train, y_test = np.load('./data.npy', allow_pickle=True)
'''

'''
plt.scatter(X_train[:,0], X_train[:,1])
plt.title('Linearly separable data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
'''
# SVM classifier
clf = svm.SVC(kernel = 'linear')
clf = clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

'''
matrix_display = ConfusionMatrixDisplay.from_estimator(
    clf,
    X_test,
    y_test,
    cmap=plt.cm.Blues,
    normalize='true'  # Normalizar los valores
)


plt.title('Confusion Matrix (Normalized)')
plt.show()
'''
support_vectors = clf.support_vectors_

plt.scatter(X_train[:,0], X_train[:,1])
plt.scatter(support_vectors[:,0], support_vectors[:,1], color = 'red')
plt.title('Support Vectors')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

plot_decision_regions(X_test, y_test, clf=clf, legend=2)
plt.show()