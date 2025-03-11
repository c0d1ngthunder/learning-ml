from sklearn import tree

features = [[35,75],[40,50],[20,45],[18,30],[35,70],[45,90],[10,30],[50,20],[10,100],[-5,100],[45,56]]
labels = [1,1,0,0,0,1,0,0,1,0,0]

AI = tree.DecisionTreeClassifier()

AI = AI.fit(features,labels)

temperature = int(input("Enter the temperature:"))
humidity = int(input("Enter the humidity:"))

if AI.predict([[temperature,humidity]]) ==1:
    print("It will rain")

else:
    print("It will not rain")