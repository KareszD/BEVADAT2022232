import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#from numba import cuda, jit

class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):

        self.root = None

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    #@jit(target_backend='cuda',forceobj=True)
    def build_tree(self, dataset, curr_depth=0):

        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split != {} and best_split["info_gain"] > 0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])

        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)

    #@jit(target_backend='cuda',forceobj=True)
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''

        best_split = {}
        max_info_gain = -float("inf")

        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        return best_split

    #@jit(target_backend='cuda',forceobj=True) #new test
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    #@jit(target_backend='cuda',forceobj=True) #new test
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "gini":
            gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain

    #@jit(target_backend='cuda')
    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    #@jit(target_backend='cuda',forceobj=True) #new
    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls ** 2
        return 1 - gini

    #@jit(target_backend='cuda',forceobj=True) #new gatya
    def calculate_leaf_value(self, Y):

        Y = list(Y)
        return max(Y, key=Y.count)

    #@jit(target_backend='cuda')
    def print_tree(self, tree=None, indent=" "):

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    #@jit(target_backend='cuda',forceobj=True) #new sehogy nem jo
    def fit(self, X, Y):

        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    #@jit(target_backend='cuda',forceobj=True)
    def predict(self, X):

        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions

    #@jit(target_backend='cuda',forceobj=True)
    def make_prediction(self, x, tree):

        if tree.value != None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor '''

        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf node
        self.value = value

from timeit import default_timer as timer

data = pd.read_csv('NJ_60k.csv')

X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values.reshape(-1,1)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.2,random_state=41)

'''
classifier = DecisionTreeClassifier(min_samples_split=3,max_depth=3)
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
print(accuracy_score(Y_test,Y_pred))


start = timer()

results = np.zeros([400,400])

for i in range(50,300,20):
    for j in range(10,50,5):
        classifier = DecisionTreeClassifier(min_samples_split=i, max_depth=j)
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
        #print(accuracy_score(Y_test, Y_pred))
        results[i][j] = round(accuracy_score(Y_test, Y_pred)*100,5)
        print(results[i][j],' i: ',i,' j: ',j)
        #print(results)
        xd = pd.DataFrame(results)
        xd.to_csv('results.csv')


print('--------------------------------------------------------------')
print(results)

print("Time:" , timer()-start)

'''
#90,10
classifier = DecisionTreeClassifier(min_samples_split=90, max_depth=10)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
print(accuracy_score(Y_test, Y_pred))


'''
Eleinte kézzel próbálkoztam reménykedve hogy közel lesz a megoldás az órai alap adatokhoz de hamar rájöttem egy két 
próba után hogy ez nem fenttartható és nem is jutottam semmire mivel az accuracy nagyon nem változott legalább is 
annyira nem hogy a 80%-hoz elég közel legyen.
Azt is észre kellett vennem hogy 6 nál feljebb nem tudott menni a max depth, először azt hittem hogy ez direkt van 
ezért kipróbáltam kb az összes értéket ezen belül de nem volt meg a 80% és ez így már egy kicsit gyanús volt
'if best_split != {} and best_split["info_gain"] > 0:' ez a sor volt a ludas. Nem volt ellenőrizve hogy üres-e a best 
split így nagyon hamar crashelt de miutáb ez abszolválva lett már bármilyen értéket meg lehetett adni de még itt sem volt 
tökéletes minden mivel rossz adathalamazon kezdtem el dolgozni :(
Sajnos a saját házim csv-jével csináltam rengetek grid-search-ös tesztet és nagyon jó eredmények jöttek ki legalább is 
azt hittem de sajnos kiderült hogy még nem volt jó a házim ezért fals adatokat kaptam de végül jött a teszt jó lett a 
házim és akkor már a jó adatokkal el tudtam kezdeni a grid-search-ölést ahol előző tapasztalatok alapján láttam hogy 
eléggé el kell távolodnom az alap értékektől és két számjegyu paraméterekkel ezek az adatok jöttek ki ahol a 90-es min_samples_split és a 10-es depth hozta a legjobb eredményt.
Viszont észre kellett vegyem hogy azért egy idő után már majdnem hogy nem változnak a százalékok mivel már túltanult a fa.

Ez a teszt 16044.311065900001 másodpercig tartott

80.10833  i:  50  j:  10
79.4  i:  50  j:  15
78.73333  i:  50  j:  20
78.54167  i:  50  j:  25
78.525  i:  50  j:  30
78.525  i:  50  j:  35
78.525  i:  50  j:  40
78.525  i:  50  j:  45
80.125  i:  70  j:  10
79.5  i:  70  j:  15
79.11667  i:  70  j:  20
79.08333  i:  70  j:  25
79.08333  i:  70  j:  30
79.08333  i:  70  j:  35
79.08333  i:  70  j:  40
79.08333  i:  70  j:  45
80.18333  i:  90  j:  10
79.70833  i:  90  j:  15
79.46667  i:  90  j:  20
79.45  i:  90  j:  25
79.45  i:  90  j:  30
79.45  i:  90  j:  35
79.45  i:  90  j:  40
79.45  i:  90  j:  45
80.14167  i:  110  j:  10
79.64167  i:  110  j:  15
79.50833  i:  110  j:  20
79.5  i:  110  j:  25
79.5  i:  110  j:  30
79.5  i:  110  j:  35
79.5  i:  110  j:  40
79.5  i:  110  j:  45
79.93333  i:  130  j:  10
79.56667  i:  130  j:  15
79.40833  i:  130  j:  20
79.4  i:  130  j:  25
79.4  i:  130  j:  30
79.4  i:  130  j:  35
79.4  i:  130  j:  40
79.4  i:  130  j:  45
79.91667  i:  150  j:  10
79.6  i:  150  j:  15
79.48333  i:  150  j:  20
79.475  i:  150  j:  25
79.475  i:  150  j:  30
79.475  i:  150  j:  35
79.475  i:  150  j:  40
79.475  i:  150  j:  45
79.90833  i:  170  j:  10
79.725  i:  170  j:  15
79.675  i:  170  j:  20
79.675  i:  170  j:  25
79.675  i:  170  j:  30
79.675  i:  170  j:  35
79.675  i:  170  j:  40
79.675  i:  170  j:  45
79.9  i:  190  j:  10
79.79167  i:  190  j:  15
79.775  i:  190  j:  20
79.775  i:  190  j:  25
79.775  i:  190  j:  30
79.775  i:  190  j:  35
79.775  i:  190  j:  40
79.775  i:  190  j:  45
79.875  i:  210  j:  10
79.74167  i:  210  j:  15
79.73333  i:  210  j:  20
79.73333  i:  210  j:  25
79.73333  i:  210  j:  30
79.73333  i:  210  j:  35
79.73333  i:  210  j:  40
79.73333  i:  210  j:  45
79.88333  i:  230  j:  10
79.75  i:  230  j:  15
79.75  i:  230  j:  20
79.75  i:  230  j:  25
79.75  i:  230  j:  30
79.75  i:  230  j:  35
79.75  i:  230  j:  40
79.75  i:  230  j:  45
79.9  i:  250  j:  10
79.775  i:  250  j:  15
79.775  i:  250  j:  20
79.775  i:  250  j:  25
79.775  i:  250  j:  30
79.775  i:  250  j:  35
79.775  i:  250  j:  40
79.775  i:  250  j:  45
79.925  i:  270  j:  10
79.80833  i:  270  j:  15
79.81667  i:  270  j:  20
79.81667  i:  270  j:  25
79.81667  i:  270  j:  30
79.81667  i:  270  j:  35
79.81667  i:  270  j:  40
79.81667  i:  270  j:  45
79.86667  i:  290  j:  10
79.76667  i:  290  j:  15
79.76667  i:  290  j:  20
79.76667  i:  290  j:  25
79.76667  i:  290  j:  30
79.76667  i:  290  j:  35
79.76667  i:  290  j:  40
79.76667  i:  290  j:  45
'''


















"""
1.  Értelmezd az adatokat!!!
    A feladat megoldásához használd a NJ transit + Amtrack csv-t a moodle-ból.
    A NJ-60k az a megoldott. Azt fogom használni a modellek teszteléséhez, illetve össze tudod hasonlítani az eredményedet.

2. Írj egy osztályt a következő feladatokra:
     2.1 Neve legyen NJCleaner és mentsd el a NJCleaner.py-ba. Ebben a fájlban csak ez az osztály legyen.
     2.2 Konsturktorban kapja meg a csv elérési útvonalát és olvassa be pandas segítségével és mentsük el a data (self.data) osztályszintű változóba
     2.3 Írj egy függvényt ami sorbarendezi a dataframe-et 'scheduled_time' szerint növekvőbe és visszatér a sorbarendezett df-el, a függvény neve legyen 'order_by_scheduled_time' és térjen vissza a df-el
     2.4 Dobjuk el a from és a to oszlopokat, illetve azokat a sorokat ahol van nan és adjuk vissza a df-et. A függvény neve legyen 'drop_columns_and_nan' és térjen vissza a df-el
     2.5 A date-et alakítsd át napokra, pl.: 2018-03-01 --> Thursday, ennek az oszlopnak legyen neve a 'day'. Ezután dobd el a 'date' oszlopot és térjen vissza a df-el. A függvény neve legyen 'convert_date_to_day' és térjen vissza a df-el
     2.6 Hozz létre egy új oszlopot 'part_of_the_day' névvel. A 'scheduled_time' oszlopból számítsd ki az alábbi értékeit. A 'scheduled_time'-ot dobd el. A függvény neve legyen 'convert_scheduled_time_to_part_of_the_day' és térjen vissza a df-el
         4:00-7:59 -- early_morning
         8:00-11:59 -- morning
         12:00-15:59 -- afternoon
         16:00-19:59 -- evening
         20:00-23:59 -- night
         0:00-3:59 -- late_night
    2.7 A késéseket jelöld az alábbiak szerint. Az új osztlop neve legyen 'delay'. A függvény neve legyen pedig 'convert_delay' és térjen vissza a df-el
         0min <= x < 5min   --> 0
         5min <= x          --> 1
    2.8 Dobd el a felesleges oszlopokat 'train_id' 'scheduled_time' 'actual_time' 'delay_minutes'. A függvény neve legyen 'drop_unnecessary_columns' és térjen vissza a df-el
    2.9 Írj egy olyan metódust, ami elmenti a dataframe első 60 000 sorát. A függvénynek egy string paramétere legyen, az pedig az, hogy hova mentse el a csv-t (pl.: 'data/NJ.csv'). A függvény neve legyen 'save_first_60k'.
    2.10 Írj egy függvényt ami a fenti függvényeket összefogja és megvalósítja (sorbarendezés --> drop_columns_and_nan --> ... --> save_first_60k), a függvény neve legyen 'prep_df'. Egy paramnétert várjon, az pedig a csv-nek a mentési útvonala legyen. Ha default value-ja legyen 'data/NJ.csv'

3.  A feladatot a HAZI06.py-ban old meg.
    Az órán megírt DecisionTreeClassifier-t fit-eld fel az első feladatban lementett csv-re.
    A feladat célja az, hogy határozzuk meg azt, hogy a vonatok késnek-e vagy sem. 0p <= x < 5p --> nem késik (0), ha 5p <= x --> késik (1).
    Az adatoknak a 20% legyen test és a splitelés random_state-je pedig 41 (mint órán)
    A testset-en 80% kell elérni. Ha megvan a minimum százalék, akkor azzal paraméterezd fel a decisiontree-t és azt kell leadni.

    A leadásnál csak egy fit kell, ezt azzal a paraméterre paraméterezd fel, amivel a legjobb accuracy-t elérted.

    A helyes paraméter megtalálásához használhatsz grid_search-öt.
    https://www.w3schools.com/python/python_ml_grid_search.asp

4.  A tanításodat foglald össze 4-5 mondatban a HAZI06.py-ban a fájl legalján kommentben. Írd le a nehézségeket, mivel próbálkoztál, mi vált be és mi nem. Ezen kívül írd le 10 fitelésed eredményét is, hogy milyen paraméterekkel probáltad és milyen accuracy-t értél el.
Ha ezt feladatot hiányzik, akkor nem fogadjuk el a házit!

HAZI-
    HAZI06-
        -NJCleaner.py
        -HAZI06.py

##################################################################
##                                                              ##
## A feladatok közül csak a NJCleaner javítom unit test-el      ##
## A decision tree-t majd manuálisan fogom lefuttatni           ##
## NJCleaner - 10p, Tanítás - acc-nál 10%-ként egy pont         ##
## Ha a 4. feladat hiányzik, akkor nem tudjuk elfogadni a házit ##
##                                                              ##
##################################################################
"""