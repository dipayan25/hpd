import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
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


#pre processing
#checking for null values
print(data.isnull().sum())
print(data.columns)
print(data.info())
print(data.head())
data.fillna(data.median())




#pre processing
X,Y=data.drop(['output'],axis=1),data['output']
print(X.shape,Y.shape)
x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.7)
print(x_train.shape,x_test.shape)


#choose an algo
outputTree=RandomForestClassifier(max_depth=2,n_estimators=80,random_state=60)


#train your model
outputTree.fit(x_train, y_train)
predTree=outputTree.predict(x_test)
train_tree=outputTree.score(x_train,y_train)
test_tree=accuracy_score(y_test, predTree)
f1_tree = f1_score(y_test, predTree, average='weighted')

#testing of model
print("Train set Accuracy: ", train_tree*100,"%")
print("Test set Accuracy: ", test_tree*100,"%")
print("Decision Tree's f1 Score: ", f1_tree*100,"%")
con=confusion_matrix(y_test,predTree)
print("\n",con)


#printing results
cm = confusion_matrix(y_test, predTree, labels=outputTree.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=outputTree.classes_)
disp.plot(cmap="Blues")
plt.title('Decision Tree Confusion matrix')
plt.show()

plt.scatter(X['age'],X['sex'],c=Y)
plt.title('GRAPH OF AGE,SEX wrt HEART DISEASES')
plt.show()

#dumping of file 
filename='heart.sav'
#pickle.dump(outputTree,open(filename,'wb'))


