from sklearn import tree

#Featuers:  0 = "bumpy" 1 = "smooth"
#Labels:    0 = apple 1 = orange

features=[[140,1],[130,1],[150,0],[170,0],[150,1],[160,1],[160,0]]
labels=[0,0,1,1,0,0,1]

clf = tree.DecisionTreeClassifier()

clf = clf.fit(features,labels)

print(clf.predict([[150,1]]))