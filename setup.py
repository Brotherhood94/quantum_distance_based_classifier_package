import setuptools

setuptools.setup(
    name="quantum_distance_based_classifier",
    version="0.0.1",
    author="Alessandro Berti",
    description="The Quantum Distance-based classifier is a technique inspired by the classical k-Nearest Neighbors that leverage quantum properties to perform prediction. The package has been implemented in Qiskit",
    packages=['quantum_distance_based_classifier'],
    url = "https://github.com/Brotherhood94/quantum_distance_based_classifier_package",
    install_requires=[
        'qiskit >= 0.44',
        'qiskit-aer >= 0.12.2',
        'numpy >= 1.25.2',
        'scikit-learn >= 1.3.0',
        'matplotlib >= 3.4.3',
        'pylatexenc >= 2.10',
    ],
)