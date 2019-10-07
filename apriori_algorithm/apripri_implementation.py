import numpy as np
import pandas as pd
from apyori import apriori
import matplotlib.pyplot as plt
import sys
from itertools import combinations, groupby
from collections import Counter

store_data = pd.read_csv('store_data.csv',header=None)
print(len(store_data))


records =[]
for i in range(7501):
    records.append([str(store_data.values[i,j]) for j in range(0,20) if j!= 'nan'])

data=[]
for row in records:
    data.append([x for x in row if x!='nan'])

for i in data:
    print(i)

association_rules = apriori(data,min_support=0.0045,min_confidence=0.2,min_lift=3,min_length=2)

# results = list(association_rules)
# for i in results:
#     print(i)

#
# print(results[0])

for item in association_rules:
    print(item)
    # first index of the inner list
    # Contains base item and add item
    pair = item[0]
    print('pair is:')
    print(pair)
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")

