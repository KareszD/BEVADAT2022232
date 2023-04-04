import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from DecisionTreeClassifier import DecisionTreeClassifier

col_name = ['sepal_length','sepal_width','petal_length','petal_width']

data = pd.read_csv('iris.csv',skiprows=1,header=None,names=col_name)

data.head(10)

X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values.reshape(-1,1)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.2,random_state=41)

classifier = DecisionTreeClassifier(min_sample_split=3,max_depth=3)
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)
print(accuracy_score(Y_test,Y_pred))
