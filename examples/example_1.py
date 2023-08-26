from quantum_distance_based_classifier.quantum_distance_based_classifier import QuantumDistaceBasedClassifier
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np


def example_1(instances_per_class=2, n_features=2):
    X, y = load_iris(return_X_y=True)
    X = X[:, :n_features] # Keep only n_features

    # Standardize and normalize the features
    X = StandardScaler().fit_transform(X)
    X = preprocessing.normalize(X, axis=1)

    # Initialize variables to store sampled instances
    sampled_X = []
    sampled_y = []

    # Loop through each class to sample instances
    for class_label in np.unique(y):
        class_indices = np.where(y == class_label)[0]
        sampled_indices = np.random.choice(class_indices, size=instances_per_class, replace=False)
        sampled_X.extend(X[sampled_indices])
        sampled_y.extend(y[sampled_indices])

    # Convert lists to numpy arrays
    sampled_X = np.array(sampled_X)
    sampled_y = np.array(sampled_y)

    # Print the sampled instances and their corresponding labels
    for i in range(len(sampled_X)):
        print(f"Instance {i+1}: Features: {sampled_X[i]}, Class: {sampled_y[i]}")
    
    qdbc = QuantumDistaceBasedClassifier()
    qdbc.fit(sampled_X, sampled_y)
    result = qdbc.predict(sampled_X[0])
    print(f"Classification result: {result}")

example_1(n_features=2)