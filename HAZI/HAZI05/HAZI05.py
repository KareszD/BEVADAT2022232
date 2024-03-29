import pandas as pd
import seaborn as sns
from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix

csv_path = "archive/iris.data.csv"

class KNNClassifier:

    def get_k(self):
        return self.k

    k_neighbors = property(fget=get_k)

    def __init__(self,k:int,test_split_ratio:float):
        self.k = k
        self.test_split_ratio = test_split_ratio

    #csv_path = "iris.csv"

    @staticmethod
    def load_csv(csv_path:str) ->Tuple[pd.core.frame.DataFrame,pd.core.frame.DataFrame]:
        dataset = pd.read_csv(csv_path,delimiter=',')
        df = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        x, y = df.iloc[:, :8], df.iloc[:, -1]
        return x, y

    '''
    x,y = load_csv(csv_path)
    x,y

    np.mean(x,axis=0),np.var(x,axis=0)

    np.nanmean(x,axis=0),np.nanvar(x,axis=0)

    x[np.isnan(x)] = 3.5
    x.shape

    np.mean(x,axis=0),np.var(x,axis=0)

    (x > 13.0).sum(), (x < 0.0).sum()

    x[np.where(np.logical_or(x > 13.0,x < 0.0))]

    less_than = np.where(x < 0.0)
    higher_than = np.where(x > 13.0)
    less_than,higher_than

    y = np.delete(y,np.where(x < 0.0)[0],axis=0)
    y = np.delete(y,np.where(x > 13.0)[0],axis=0)
    x = np.delete(x,np.where(x < 0.0)[0],axis=0)
    x = np.delete(x,np.where(x > 13.0)[0],axis=0)
    x.shape,y.shape
    '''

    def train_test_split(self,features:pd.core.frame.DataFrame,labels:pd.core.frame.DataFrame) -> None:

        test_size = int(len(features) * self.test_split_ratio)
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size, "Size mismatch!"

        x_train, y_train = features[:train_size], labels[:train_size]
        x_test, y_test = features[train_size:], labels[train_size:]
        self.x_train = x_train.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)
        self.x_test = x_test.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        #return (x_train,y_train,x_test,y_test)

    def euclidean(self,element_of_x:pd.core.frame.Series) -> pd.core.frame.DataFrame:
        eu = self.x_train-element_of_x
        eu = eu ** 2
        eu = eu.sum(axis=1)
        eu = eu ** (1/2)
        return eu

        #return np.sqrt(np.sum((self.x_train - element_of_x)**2,axis=1))

    def predict(self, x_test):
        labels_pred = []
        for i in range(len(x_test)):
            distances = KNNClassifier.euclidean(self,x_test.iloc[i]).tolist()
            for j in range(len(distances)):
                distances[j] = (distances[j],self.y_train[j])
            distances.sort(key=lambda a: a[0])
            distances = distances[:self.k]
            for h in range(len(distances)):
                distances[h] = distances[h][1]
            labels_pred.append(mode(distances, keepdims=False).mode.item())
            #print(mode(distances))


        #print(labels_pred)
        self.y_preds = pd.Series(labels_pred)

    def accuracy(self) -> float:
        true_positive = (self.y_test == self.y_preds).sum()
        return true_positive / len(self.y_test) * 100

    def confusion_matrix(self):
            conf_matrix = confusion_matrix(self.y_test,self.y_preds)
            #sns.heatmap(conf_matrix,annot=True)
            return conf_matrix

    def best_k(self)->Tuple[int,float]:
        results = list()
        for i in range(1, 20):
            self.k = i
            KNNClassifier.predict(self,self.x_test)
            acc = round(KNNClassifier.accuracy(self), 2)
            results.append((i, acc))

        return max(results, key=lambda item: item[1])

#x, y = KNNClassifier.load_csv("diabetes.csv")

#classifier = KNNClassifier(4, 0.2)

#classifier.train_test_split(x, y)
#classifier.predict(classifier.x_test)

#print(classifier.accuracy())
#classifier.confusion_matrix()

#print(classifier.best_k())