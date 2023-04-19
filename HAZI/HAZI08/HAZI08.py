import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
'''
Készíts egy függvényt, betölti majd vissza adja az iris adathalmazt.


Egy példa a kimenetre: iris
return type: sklearn.utils.Bunch
függvény neve: load_iris_data
'''

def load_iris_data() -> sklearn.utils.Bunch:
    return load_iris()

#print(load_iris_data())

'''
Készíts egy függvényt, ami a betölti az virágokhoz tartozó levél méretket egy dataframebe, majd az első 5 sort visszaadja.
Minden oszlop tartalmazza, hogy az milyen mérethez tartozik.

Egy példa a bemenetre: iris
Egy példa a kimenetre: iris_df
return type: pandas.core.frame.DataFrame
függvény neve: check_data
'''

def check_data(iris)->pd.core.frame.DataFrame:
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    return df.iloc[:5,:2]

#print(check_data(load_iris_data()))

''' 
Készíts egy függvényt ami előkészíti az adatokat egy lineaáris regressziós model feltanításához.
Featurejeink legyenek a levél méretek kivéve a "sepal length (cm)", ez legyen a targetünk.

Egy példa a bemenetre: iris
Egy példa a kimenetre: X, y
return type: (numpy.ndarray, numpy.ndarray)
'''

def preplinreg(iris):
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    X = df["sepal width (cm)"].values.reshape(-1,1)
    y = df['sepal length (cm)'].values.reshape(-1,1)
    return X,y

''' 
Készíts egy függvényt ami előkészíti az adatokat egy logisztikus regressziós model feltanításához.
Featurejeink legyenek a levél méretek, targetünk pedig a 0, 1-es virág osztályok.
Fontos csak azokkal az adatokkal tanítsunk amihez tartozik adott target. 

Egy példa a bemenetre: iris
Egy példa a kimenetre: X, y
return type: (numpy.ndarray, numpy.ndarray)
'''

def preplogreg(iris):
    df = pd.DataFrame(iris.data)
    df['class'] = iris.target

    df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

'''
Készíts egy függvényt ami feldarabolja az adatainkat train és test részre. Az adatok 20%-át használjuk fel a teszteléshez.
Tegyük determenisztikussá a darabolást, ennek értéke legyen 42.

Egy példa a bemenetre: X, y
Egy példa a kimenetre: X_train, X_test, y_train, y_test
return type: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
'''

def dara(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

'''
Készíts egy függvényt ami feltanít egy lineaáris regressziós modelt, majd visszatér vele.

Egy példa a bemenetre: X_train, y_train
Egy példa a kimenetre: model
return type: sklearn.linear_model._base.LinearRegression
'''

def fitlin(X_train,y_train)->LinearRegression:
    a = LinearRegression()
    a.fit(X = X_train,y = y_train)
    return a

'''
Készíts egy függvényt ami feltanít egy logisztikus regressziós modelt, majd visszatér vele.

Egy példa a bemenetre: X_train, y_train
Egy példa a kimenetre: model
return type: sklearn.linear_model._base.LogisticRegression
'''

def fitlog(X_train,y_train)->LogisticRegression:
    return LogisticRegression().fit(X = X_train,y = y_train)

''' 
Készíts egy függvényt, ami a feltanított modellel predikciót tud végre hajtani.

Egy példa a bemenetre: model, X_test
Egy példa a kimenetre: y_pred
return type: numpy.ndarray
'''

def pred(model,X_test)->numpy.ndarray:
    if model is LogisticRegression:
        return model.predict(X = X_test)
    else:
        return model.predict(X = X_test)

'''
Készíts egy függvényt, ami vizualizálni tudja a label és a predikciók közötti eltérést.
Használj scatter plotot a diagram elkészítéséhez.

Diagram címe legyen: 'Actual vs Predicted Target Values'
Az x tengely címe legyen: 'Actual'
Az y tengely címe legyen: 'Predicted'

Egy példa a bemenetre: y_test, y_pred
Egy példa a kimenetre: scatter plot
return type: matplotlib.figure.Figure
'''

def plotos(y_test,y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_title('Actual vs Predicted Target Values')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    return fig

''' 
Készíts egy függvényt, ami a Négyzetes hiba (MSE) értékét számolja ki a predikciók és a valós értékek között.

Egy példa a bemenetre: y_test, y_pred
Egy példa a kimenetre: mse
return type: float
'''

def mse(y_test,y_pred):
    return np.mean((y_pred - y_test)**2)
'''
#LIN
iris = load_iris_data()
X,y = preplinreg(iris)
X_train, X_test, y_train, y_test = dara(X,y)
linmod = fitlin(X_train,y_train)
preds = pred(linmod,X_test)
plotos(y_test,preds)
plt.show()
print(mse(y_test,preds))
'''
'''
#LOG
iris = load_iris_data()
X,y = preplogreg(iris)
X_train, X_test, y_train, y_test = dara(X,y)
linmod = fitlog(X_train,y_train)
preds = pred(linmod,X_test)
plotos(y_test,preds)
plt.show()
print(mse(y_test,preds))
'''


