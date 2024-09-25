import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Adaline import Adaline
from sklearn.svm import SVC


# Generació del conjunt de mostres
X, y = make_classification(n_samples=400, n_features=5, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1,
                           random_state=9)

# Separar les dades: train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Estandaritzar les dades: StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenam una SVM linear (classe SVC)
svm = SVC(C=1000.0, kernel='linear') # C és el paràmetre de regularització, kernel és el tipus de kernel
svm.fit(X_train, y_train) # Entrenament SVM


# Prediccio
y_prediction_svm = svm.predict(X_test) # Prediccio SVM

# Metrica
num_correctes = np.sum(y_prediction_svm == y_test)