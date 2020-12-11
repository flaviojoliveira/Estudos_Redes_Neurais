import numpy as np
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

np.random.seed(8)

iris = datasets.load_iris()
x = iris.data
y = iris.target

scaler.fit(x)
x = scaler.transform(x)

sufInd = np.arange(150)
np.random.shuffle(sufInd)

x_train = x[sufInd[:100], :]
x_test  = x[sufInd[100:],:]
y_train = y[sufInd[:100]]
y_test  = y[sufInd[100:]]

clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(4, earning_rate_init=0.01, activation='logistic', max iter=1500, random_state=1)
clf.fit(x_train, y_train)
y_pred = clf.predict_proba(x_test)
y_aux = np.argmax(y_pred, 1)
accuracy_score(y_test, y_aux)

print(y_test[:10], y_aux[:10])
