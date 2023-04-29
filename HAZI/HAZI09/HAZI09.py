# %%
# imports
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets

sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.metrics import confusion_matrix

# %%
# Készíts egy függvényt ami betölti a digits datasetet
# NOTE: használd az sklearn load_digits-et
# Függvény neve: load_digits()
# Függvény visszatérési értéke: a load_digits visszatérési értéke


# %%
# Vizsgáld meg a betöltött adatszetet (milyen elemek vannak benne stb.)


# %%
# Vizsgáld meg a data paraméterét a digits dataset-nek (tartalom,shape...)


# %%
# Készíts egy függvényt ami létrehoz egy KMeans model-t 10 db cluster-el
# NOTE: használd az sklearn Kmeans model-jét (random_state legyen 0)
# Miután megvan a model predict-elj vele
# NOTE: használd a fit_predict-et
# Függvény neve: predict(n_clusters:int,random_state:int,digits)
# Függvény visszatérési értéke: (model:sklearn.cluster.KMeans,clusters:np.ndarray)



# %%
# Vizsgáld meg a shape-jét a kapott model cluster_centers_ paraméterének.


# %%
# Készíts egy plotot ami a cluster középpontokat megjeleníti


# %%
# Készíts egy függvényt ami visszaadja a predictált cluster osztályokat
# NOTE: amit a predict-ből visszakaptunk "clusters" azok lesznek a predictált cluster osztályok
# HELP: amit a model predictált cluster osztályok még nem a labelek, hanem csak random cluster osztályok,
#       Hogy label legyen belőlük:
#       1. készíts egy result array-t ami ugyan annyi elemű mint a predictált cluster array
#       2. menj végig mindegyik cluster osztályon (0,1....9)
#       3. készíts egy maszkot ami az adott cluster osztályba tartozó elemeket adja vissza
#       4. a digits.target-jét indexeld meg ezzel a maszkkal
#       5. számold ki ennel a subarray-nek a móduszát
#       6. a result array-ben tedd egyenlővé a módusszal azokat az indexeket ahol a maszk True
#       Erre azért van szükség mert semmi nem biztosítja nekünk azt, hogy a "0" cluster a "0" label lesz, lehet, hogy az "5" label lenne az.

# Függvény neve: get_labels(clusters:np.ndarray, digits)
# Függvény visszatérési értéke: labels:np.ndarray


# %%
# Készíts egy függvényt ami kiszámolja a model accuracy-jét
# Függvény neve: calc_accuracy(target_labels:np.ndarray,predicted_labels:np.ndarray)
# Függvény visszatérési értéke: accuracy:float
# NOTE: Kerekítsd 2 tizedes jegyre az accuracy-t

# %%
# Készíts egy confusion mátrixot és plot-old seaborn segítségével

class KMeansOnDigits():
    def __init__(self, n_clusters=10, random_state=0):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def load_dataset(self):
        self.digits = sklearn.datasets.load_digits()

    def predict(self):
        kmeans = KMeans(n_clusters=self.n_clusters,random_state=self.random_state)
        X = self.digits.data
        y = self.digits.target
        #kmeans.fit_predict(X=X,y=y)
        self.clusters = kmeans.fit_predict(X=X,y=y)
        #self.xd = kmeans.labels_

    def get_labels(self):
        result = np.empty(self.clusters.shape)
        for x in self.digits.target_names:
            mask = self.clusters == x
            sub_arr = self.digits.target[mask]
            arr_mod = mode(sub_arr).mode.item()
            result[mask] = arr_mod
        self.labels = result

        #self.labels = self.xd

    def calc_accuracy(self):
        self.accuracy = np.round(accuracy_score(self.digits.target,self.labels),2)

    def confusion_matrix(self):
        self.mat = confusion_matrix(self.digits.target,self.labels)

'''
xd = KMeansOnDigits()
xd.load_dataset()
xd.predict()
xd.get_labels()
xd.calc_accuracy()
xd.confusion_matrix()
print(xd.accuracy)
print(xd.labels)
'''