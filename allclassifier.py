import pandas as pd

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_excel('final_feat.xlsx',header=None)
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Support Vector Machine
## 2. SVM (linear)
clf = svm.SVC(kernel = 'rbf')
clf.fit(X_train, y_train)

output = clf.predict(X_test)
smv_rbf = accuracy_score(y_test, output)
print('RBF SVM=',smv_rbf)

clf3 = svm.SVC(kernel = 'poly')
clf3.fit(X_train, y_train)
output = clf3.predict(X_test)
poly = accuracy_score(y_test, output)
print('poly SVM=',poly)

clf4 = svm.SVC(kernel = 'sigmoid')
clf4.fit(X_train, y_train)
output = clf4.predict(X_test)
sigmoid = accuracy_score(y_test, output)
print('sigmoid SVM=',sigmoid)

# Gaussian Neive Bayes
clf5 = GaussianNB()
clf5.fit(X_train, y_train)
output = clf5.predict(X_test)
NB = accuracy_score(y_test, output)
print('NB=',NB)

#Decision Tree
clf6 = DecisionTreeClassifier() 
clf6.fit(X_train, y_train)
output = clf6.predict(X_test)
DT = accuracy_score(y_test, output)
print('DT=',DT)

# K nearest Neighbors
clf7=KNeighborsClassifier(n_neighbors=3)
clf7.fit(X_train, y_train)
output = clf7.predict(X_test)
knn = accuracy_score(y_test, output)
print('KNN3=',knn)

clf7=KNeighborsClassifier(n_neighbors=5)
clf7.fit(X_train, y_train)
output = clf7.predict(X_test)
knn = accuracy_score(y_test, output)
print('KNN5=',knn)

clf7=KNeighborsClassifier(n_neighbors=7)
clf7.fit(X_train, y_train)
output = clf7.predict(X_test)
knn = accuracy_score(y_test, output)
print('KNN7=',knn)

# stochastic gradient descent (SGD)
clf8=SGDClassifier()
clf8.fit(X_train, y_train)
output = clf8.predict(X_test)
SGD = accuracy_score(y_test, output)
print('SGD=',SGD)

# Gradient Boosting
clf9=GradientBoostingClassifier(n_estimators=100)
clf9.fit(X_train, y_train)
output = clf9.predict(X_test)
GBC = accuracy_score(y_test, output)
print('GBC=',GBC)

# Multi Layer Perceptron
clf10=MLPClassifier()
clf10.fit(X_train, y_train)
output = clf10.predict(X_test)
MLP = accuracy_score(y_test, output)
print('MLP=',MLP)

# K-Mean clusturing
clf12=KMeans()
clf12.fit(X_train, y_train)
output = clf12.predict(X_test)
kmean = accuracy_score(y_test, output)
print('kmean=',kmean)

# Adaboost Classifier
clf13=AdaBoostClassifier()
clf13.fit(X_train, y_train)
output = clf13.predict(X_test)
Adaboost = accuracy_score(y_test, output)
print('Adaboost=',Adaboost)
