import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
import pickle 
from sklearn import metrics
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

#import warning
import warnings
warnings.filterwarnings("ignore")

#data gathering
data=pd.read_csv("heartproject/heart.csv")
print(data.isnull().sum())
print(data.columns)
print(data.info())
print(data.head())
data.fillna(data.median())

X,Y=data.drop(['output'],axis=1),data['output']
print(X.shape,Y.shape)
x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.8)
print(x_train.shape,x_test.shape)

print("RandomForest Classifier Algorithm")
outputTree=RandomForestClassifier(max_depth=2,n_estimators=60,random_state=30)
outputTree.fit(x_train, y_train)
predTree=outputTree.predict(x_test)
train_tree=outputTree.score(x_train,y_train)
test_tree=accuracy_score(y_test, predTree)
f1_tree = f1_score(y_test, predTree, average='weighted')

print("Train set Accuracy: ", train_tree*100,"%")
print("Test set Accuracy: ", test_tree*100,"%")
print("RandomForest  f1 Score: ", f1_tree*100,"%")
con=confusion_matrix(y_test,predTree)
plt.bar(1,test_tree)
#plt.subplot(2,3,1)
print("\n",con)

print("")

print("SVM Algorithm")

outputTree=SVC()
outputTree.fit(x_train, y_train)
predTree=outputTree.predict(x_test)
train_tree=outputTree.score(x_train,y_train)
test_tree=accuracy_score(y_test, predTree)
f1_tree = f1_score(y_test, predTree, average='weighted')

print("Train set Accuracy: ", train_tree*100,"%")
print("Test set Accuracy: ", test_tree*100,"%")
print("SVM Algorithm f1 Score: ", f1_tree*100,"%")
con=confusion_matrix(y_test,predTree)
print("\n",con)
plt.bar(2,test_tree)
#plt.subplot(2,3,2)

print("")


print("Decision Tree")

outputTree=DecisionTreeClassifier()
outputTree.fit(x_train, y_train)
predTree=outputTree.predict(x_test)
train_tree=outputTree.score(x_train,y_train)
test_tree=accuracy_score(y_test, predTree)
f1_tree = f1_score(y_test, predTree, average='weighted')

print("Train set Accuracy: ", train_tree*100,"%")
print("Test set Accuracy: ", test_tree*100,"%")
print("Decision Tree's f1 Score: ", f1_tree*100,"%")
con=confusion_matrix(y_test,predTree)
print("\n",con)
plt.bar(3,test_tree)
#plt.subplot(2,3,3)

print("")


print("Logistic Regression")

outputTree=LogisticRegression()
outputTree.fit(x_train, y_train)
predTree=outputTree.predict(x_test)
train_tree=outputTree.score(x_train,y_train)
test_tree=accuracy_score(y_test, predTree)
f1_tree = f1_score(y_test, predTree, average='weighted')

print("Train set Accuracy: ", train_tree*100,"%")
print("Test set Accuracy: ", test_tree*100,"%")
print("Logistic Regression f1 Score: ", f1_tree*100,"%")
con=confusion_matrix(y_test,predTree)
print("\n",con)
plt.bar(4,test_tree)
#plt.subplot(2,3,4)

print("")

print("Naive Basis Classification")

outputTree=GaussianNB()
outputTree.fit(x_train, y_train)
predTree=outputTree.predict(x_test)
train_tree=outputTree.score(x_train,y_train)
test_tree=accuracy_score(y_test, predTree)
f1_tree = f1_score(y_test, predTree, average='weighted')

print("Train set Accuracy: ", train_tree*100,"%")
print("Test set Accuracy: ", test_tree*100,"%")
print("Naive Basis Classification f1 Score: ", f1_tree*100,"%")
con=confusion_matrix(y_test,predTree)
print("\n",con)
plt.bar(5,test_tree)


print("")

print("KNN Algorithm")
outputTree=KNeighborsClassifier()
outputTree.fit(x_train, y_train)
predTree=outputTree.predict(x_test)
train_tree=outputTree.score(x_train,y_train)
test_tree=accuracy_score(y_test, predTree)
f1_tree = f1_score(y_test, predTree, average='weighted')

print("Train set Accuracy: ", train_tree*100,"%")
print("Test set Accuracy: ", test_tree*100,"%")
print("KNN Algorithm f1 Score: ", f1_tree*100,"%")
con=confusion_matrix(y_test,predTree)
print("\n",con)
plt.plot(6,test_tree)

plt.xlabel("Acurracy")
plt.ylabel("Models")
plt.title("Testing Accuracy graph of Different Model  ")
model_name={'1:RandomForest','2:SVM','3:Decision Tree','4:Logistic Regression','5:Naive Basis Classification','6:KNN Algorithm'}
plt.legend(model_name)
plt.show()
print("")





