import math
import random

import pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#pd.set_option('display.max_rows',None)
#pd.set_option('display.max_columns',None)

'''
FONTOS: Az első feladatáltal visszaadott DataFrame-et kell használni a további feladatokhoz. 
A függvényeken belül mindig készíts egy másolatot a bemenő df-ről, (new_df = df.copy() és ezzel dolgozz tovább.)
'''


'''
Készíts egy függvényt, ami egy string útvonalat vár paraméterként, és egy DataFrame ad visszatérési értékként.

Egy példa a bemenetre: 'test_data.csv'
Egy példa a kimenetre: df_data
return type: pandas.core.frame.DataFrame
függvény neve: csv_to_df
'''

def csv_to_df(datas):
    return pd.read_csv(datas)

df = csv_to_df('StudentsPerformance.csv')
#print(df)




'''
Készíts egy függvényt, ami egy DataFrame-et vár paraméterként, 
és átalakítja azoknak az oszlopoknak a nevét nagybetűsre amelyiknek neve nem tartalmaz 'e' betüt.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_capitalized
return type: pandas.core.frame.DataFrame
függvény neve: capitalize_columns
'''

def capitalize_columns(df1):
    newdf = df1.copy()
    tmplist = []
    for x in newdf.columns:
        if 'e' in x:
            tmplist.append(x.upper())
        else:
            tmplist.append(x)
    newdf.columns = tmplist
    return newdf

#print(capitalize_columns(df))



'''
Készíts egy függvényt, ahol egy szám formájában vissza adjuk, hogy hány darab diáknak sikerült teljesíteni a matek vizsgát.
(legyen az átmenő ponthatár 50).

Egy példa a bemenetre: df_data
Egy példa a kimenetre: 5
return type: int
függvény neve: math_passed_count
'''

def math_passed_count(df1):
    newdf = df1.copy()
    columos = newdf['math score'].astype(int)
    return columos[columos >= 50].count()

#print(math_passed_count(df))

'''
Készíts egy függvényt, ahol Dataframe ként vissza adjuk azoknak a diákoknak az adatait (sorokat), akik végeztek előzetes gyakorló kurzust.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_did_pre_course
return type: pandas.core.frame.DataFrame
függvény neve: did_pre_course
'''

def did_pre_course(df1):
    newdf = df1.copy()
    return newdf.loc[newdf['test preparation course'] == 'completed']

#print(did_pre_course(df))



'''
Készíts egy függvényt, ahol a bemeneti Dataframet a diákok szülei végzettségi szintjei alapján csoportosításra kerül,
majd aggregációként vegyük, hogy átlagosan milyen pontszámot értek el a diákok a vizsgákon.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_average_scores
return type: pandas.core.frame.DataFrame
függvény neve: average_scores
'''

def average_scores(df1):
    newdf = df1.copy()
    score = ['math score','reading score','writing score']
    return newdf.groupby("parental level of education")[score].mean()

#print(average_scores(df))



'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'age' oszloppal, töltsük fel random 18-66 év közötti értékekkel.
A random.randint() függvényt használd, a random sorsolás legyen seedleve, ennek értéke legyen 42.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_age
return type: pandas.core.frame.DataFrame
függvény neve: add_age
'''
def add_age(df1):
    newdf = df1.copy()
    random.seed(42)
    newdf["age"] = 0
    for x in range(len(newdf["age"])):
        newdf["age"][x] = random.randint(18,66)
    return newdf

#print(add_age(df))

'''
Készíts egy függvényt, ami vissza adja a legjobb teljesítményt elérő női diák pontszámait.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: (99,99,99) #math score, reading score, writing score
return type: tuple
függvény neve: female_top_score
'''

def female_top_score(df1):
    newdf = df1.copy()
    females = newdf.loc[newdf['gender'] == 'female']
    score = ['math score', 'reading score', 'writing score']
    females['avg'] = females[score].mean(axis=1)
    females = pd.DataFrame.sort_values(females, by=['avg'])[::-1]
    mytuple = (females.iloc[0]['math score'],females.iloc[0]['reading score'],females.iloc[0]['writing score'])
    return mytuple

#print(female_top_score(df))

'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'grade' oszloppal. 
Számoljuk ki hogy a diákok hány százalékot ((math+reading+writing)/300) értek el a vizsgán, és osztályozzuk őket az alábbi szempontok szerint:

90-100%: A
80-90%: B
70-80%: C
60-70%: D
<60%: F

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_grade
return type: pandas.core.frame.DataFrame
függvény neve: add_grade
'''

def add_grade(df1):
    newdf = df1.copy()
    score = ['math score', 'reading score', 'writing score']
    newdf['avg'] = newdf[score].mean(axis=1)
    newdf["grade"] = ""
    for x in range(len(newdf["avg"])):
        result = newdf['avg'][x]
        if 90 <= result <=100:
            newdf['grade'][x] = 'A'
        elif 80 <= result <90:
            newdf['grade'][x] = 'B'
        elif 70 <= result <80:
            newdf['grade'][x] = 'C'
        elif 60 <= result <70:
            newdf['grade'][x] = 'D'
        else:
            newdf['grade'][x] = 'E'
    return newdf

#print(add_grade(df))

'''
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan oszlop diagrammot,
ami vizualizálja a nemek által elért átlagos matek pontszámot.

Oszlopdiagram címe legyen: 'Average Math Score by Gender'
Az x tengely címe legyen: 'Gender'
Az y tengely címe legyen: 'Math Score'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: math_bar_plot
'''





''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan histogramot,
ami vizualizálja az elért írásbeli pontszámokat.

A histogram címe legyen: 'Distribution of Writing Scores'
Az x tengely címe legyen: 'Writing Score'
Az y tengely címe legyen: 'Number of Students'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: writing_hist
'''





''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan kördiagramot,
ami vizualizálja a diákok etnikum csoportok szerinti eloszlását százalékosan.

Érdemes megszámolni a diákok számát, etnikum csoportonként,majd a százalékos kirajzolást az autopct='%1.1f%%' paraméterrel megadható.
Mindegyik kör szelethez tartozzon egy címke, ami a csoport nevét tartalmazza.
A diagram címe legyen: 'Proportion of Students by Race/Ethnicity'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: ethnicity_pie_chart
'''