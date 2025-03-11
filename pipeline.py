from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = .1)

# from sklearn import tree
# clf = tree.DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()

clf.fit(X_train,y_train)

print(clf.predict(X_test))
print(accuracy_score(y_test,clf.predict(X_test)))