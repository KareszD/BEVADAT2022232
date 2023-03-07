import time
from datetime import date, timedelta, datetime

import numpy as np

# Írj egy olyan fügvényt, ami megfordítja egy 2d array oszlopait
# Be: [[1,2],[3,4]]
# Ki: [[2,1],[4,3]]
# column_swap()

def column_swap(array):
    return np.roll(array, 1, 1)

#print(column_swap([[1,2],[3,4]]))

#Készíts egy olyan függvényt ami összehasonlít két array-t és adjon vissza egy array-ben, hogy hol egyenlőek
# Pl Be: [7,8,9], [9,8,7]
# Ki: [1]
# compare_two_array()
# egyenlő elemszámúakra kell csak hogy működjön

def compare_two_array(array1,array2):
    a = np.array(array1) == np.array(array2)
    b = []
    for i in range(len(a)):
        if a[i] == True:
            b.append(i)
    return b

#print(compare_two_array([7,8,9,10], [9,8,7,10]))

# Készíts egy olyan függvényt, ami vissza adja a megadott array dimenzióit:
# Be: [[1,2,3], [4,5,6]]
# Ki: "sor: 2, oszlop: 3, melyseg: 1"
# get_array_shape()
# 3D-vel még műküdnie kell!

def get_array_shape(array):
    a = np.array(array)
    output = ''
    #sor
    if len(a.shape) > 0:
        output += f'sor: {a.shape[0]}, '
    else:
        output += f'sor: 1, '
    #oszlop
    if len(a.shape) > 1:
        output += f'oszlop: {a.shape[1]}, '
    else:
        output += f'oszlop: 1, '
    #melyseg
    if len(a.shape) > 2:
        output += f'melyseg: {a.shape[2]}'
    else:
        output += f'melyseg: 1'
    return output

#print(get_array_shape([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])) #3D
#print(get_array_shape([[1,2,3], [4,5,6]])) #2D

# Készíts egy olyan függvényt, aminek segítségével elő tudod állítani egy neurális hálózat tanításához szükséges Y-okat egy numpy array-ből.
#Bementként add meg az array-t, illetve hogy mennyi class-od van. Kimenetként pedig adjon vissza egy 2d array-t, ahol a sorok az egyes elemek. Minden nullákkal teli legyen és csak ott álljon egyes, ahol a bementi tömb megjelöli
# Be: [1, 2, 0, 3], 4
# Ki: [[0,1,0,0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
# encode_Y()

def encode_Y(array,soros):
    a = []
    for i in range(soros):
        b = np.zeros(soros)
        b[array[i]] = 1
        a.append(b)
    return a

#print(encode_Y([1, 2, 0, 3], 4))

# A fenti feladatnak valósítsd meg a kiértékelését. Adj meg a 2d array-t és adj vissza a decodolt változatát
# Be:  [[0,1,0,0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
# Ki:  [1, 2, 0, 3]
# decode_Y()

def decode_Y(array):
    array = np.array(array)
    a = []
    for i in array:
        a.append(np.where(i == 1))
    a = np.array(a,int)
    return a.tolist()

#print(decode_Y([[0,1,0,0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]))

# Készíts egy olyan függvényt, ami képes kiértékelni egy neurális háló eredményét! Bemenetként egy listát és egy array-t és adja vissza a legvalószínübb element a listából.
# Be: ['alma', 'körte', 'szilva'], [0.2, 0.2, 0.6].
# Ki: 'szilva'
# eval_classification()

def eval_classification(array1,array2):
    idx = np.argmax(array2,0)
    return array1[idx]

#print(eval_classification(['alma', 'körte', 'szilva'], [0.2, 0.2, 0.6]))

# Készíts egy olyan függvényt, ahol az 1D array-ben a páratlan számokat -1-re cseréli
# Be: [1,2,3,4,5,6]
# Ki: [-1,2,-1,4,-1,6]
# replace_odd_numbers()

def replace_odd_numbers(array):
    a = np.array(array)
    return np.where(a%2==1,-1,a)

#print(replace_odd_numbers([1,2,3,4,5,6]))

# Készíts egy olyan függvényt, ami egy array értékeit -1 és 1-re változtatja, attól függően, hogy az adott elem nagyobb vagy kisebb a paraméterként megadott számnál.
# Ha a szám kisebb mint a megadott érték, akkor -1, ha nagyobb vagy egyenlő, akkor pedig 1.
# Be: [1, 2, 5, 0], 2
# Ki: [-1, 1, 1, -1]
# replace_by_value()

def replace_by_value(array,num):
    array = np.array(array)
    array[array<num] = -1
    array[array>=num] = 1
    return array

#print(replace_by_value([1, 2, 5, 0], 2))

# Készítsd egy olyan függvényt, ami az array értékeit összeszorozza és az eredmény visszaadja
# Be: [1,2,3,4]
# Ki: 24
# array_multi()
# Ha több dimenziós a tömb, akkor az egész tömb elemeinek szorzatával térjen vissza

def array_multi(array):
    return np.prod(array)

#print(array_multi([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

# Készítsd egy olyan függvényt, ami a 2D array értékeit összeszorozza és egy olyan array-el tér vissza, aminek az elemei a soroknak a szorzata
# Be: [[1, 2], [3, 4]]
# Ki: [2, 12]
# array_multi_2d()

def array_multi_2d(array):
    a = []
    for i in array:
        a.append(np.prod(i))
    return a

#print(array_multi_2d([[1, 2], [3, 4]]))

# Készíts egy olyan függvényt, amit egy meglévő numpy array-hez készít egy bordert nullásokkal. Bementként egy array-t várjon és kimenetként egy array jelenjen meg aminek van border-je
# Be: [[1,2],[3,4]]
# Ki: [[0,0,0,0],[0,1,2,0],[0,3,4,0],[0,0,0,0]]
# add_border()

def add_border(innerarray):
    array = np.pad(innerarray, pad_width=1, mode='constant',constant_values=0)
    return array.tolist()

#print(add_border([[1,2],[3,4]]))

# Készíts egy olyan függvényt ami két dátum között felsorolja az összes napot.
# Be: '2023-03', '2023-04'
# Ki: ['2023-03-01', '2023-03-02', .. , '2023-03-31',]
# list_days()

def list_days(date1,date2):
    a = []
    start = np.datetime64(date1,'D')
    end = np.datetime64(date2,'D')
    delta = np.timedelta64(1, 'D')
    days = np.arange(start,end,delta)
    for i in days:
        a.append(str(i))
    return a

#print(list_days('2023-03', '2023-04'))

# Írj egy fügvényt ami vissza adja az aktuális dátumot az alábbi formában: YYYY-MM-DD
# Be:
# Ki: 2017-03-24

def current_date():
    return date.today()

#print(current_date())

# Írj egy olyan függvényt ami visszadja, hogy mennyi másodperc telt el 1970 január 01. 00:00:00 óta.
# Be:
# Ki: másodpercben az idó, int-é kasztolva
# sec_from_1970()

def sec_from_1970():
    return int(time.time())

#print(sec_from_1970())