import setuptools

setuptools.setup(
    name="quantum_distance_based_classifier",
    version="0.0.4",
    author="Alessandro Berti",
    description="The Quantum Distance-based classifier is a technique inspired by the classical k-Nearest Neighbors that leverage quantum properties to perform prediction. The package has been implemented in Qiskit",
    packages=['quantum_distance_based_classifier'],
    url = "https://github.com/Brotherhood94/quantum_distance_based_classifier_package",
    install_requires=[
        'qiskit >= 0.44',
        'qiskit-aer >= 0.12.2',
        'numpy >= 1.24.3',
        'scikit-learn >= 1.2.2',
        'matplotlib >= 3.4.3',
        'pylatexenc >= 2.10',
    ],
    long_description="""The Quantum Distance-based classifier is a technique inspired by the classical k-Nearest Neighbors that leverage quantum properties to perform prediction. The package has been implemented in Qiskit.\n
        ```
        from quantum_distance_based_classifier.quantum_distance_based_classifier import QuantumDistaceBasedClassifier
        from sklearn import preprocessing
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler
        import numpy as np


        X, y = load_iris(return_X_y=True)

        n_features = 2
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

        qdbc = QuantumDistaceBasedClassifier()
        qdbc.fit(sampled_X, sampled_y)
        result = qdbc.predict(sampled_X[0])
        print(f"Classification result: {result}")
        ```
        """,
    long_description_content_type='text/markdown',
)