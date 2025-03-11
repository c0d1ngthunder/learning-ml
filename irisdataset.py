from sklearn.datasets import load_iris
import numpy as n
from sklearn import tree

iris=load_iris()

test_index=[0,45,51,101]

train_trg=n.delete(iris.target,test_index)
train_data=n.delete(iris.data,test_index,axis=0)


# this must be the answer from the AI 
test_target = iris.target[test_index]

# this data will be used to test the AI
test_data = iris.data[test_index]

ai = tree.DecisionTreeClassifier()

ai.fit(train_data,train_trg)

print(test_target)
print(ai.predict(test_data))