import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random
from sklearn.model_selection import train_test_split

# below is the actual algorithm code
def k_nearest_neighbors(data,predict,k=3):
    # since we want the number of neighbors that we are looking at to be informative enough
    # we don't want k to be just equal or even less than the number of groups we have in our dataset
    if len(data)>=k:
        warnings.warn('K is set to a value less than total voting groups!')

    distance=[]
    # since while calculating each data point's distance away from the point of interest, we also want to record that which class(group)
    # that this each point belongs to (so we can count the final vote), we need to put a group label on each distance calculated
    # the best way to do this is to iterate through groups(以组为单位) while accessing each point
    for group in data:
        # go through the features of each data point within the group
        for features in data[group]:
            # 这两个matrix相减所得matrix的norm即我们想要计算的distance(其实就是多维勾股定理)
            euclidean_distance= np.linalg.norm(np.array(features)-np.array(predict))
            # append the calculated distance with the group that this point belongs to together as a list into the distance list
            distance.append([euclidean_distance,group])
    # get the top k shortest distance
    votes = [i[1] for i in sorted(distance)[:k]]
    # count which group appears the most number of times in the top k as the final classification result
    # Counter(votes).most_common(1) returns a list of tuples, where the first element in the first tuple is the "most voted" class
    vote_result=Counter(votes).most_common(1)[0][0]
    confidance = Counter(votes).most_common(1)[0][1]/k

    print("Vote result is: ",vote_result,"with the confidance of ",confidance)
    return vote_result,confidance

df = pd.read_csv("breast-cancer-wisconsin.data")
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
# make sure all the data entries are float and convert the table to a list of lists where each small list represent all the features of one data point
full_data=df.astype(float).values.tolist()
random.shuffle(full_data)

test_size=0.2
# we make the keys of the following dicts as 2 and 4 since in the dataset, 2 means benign and 4 means malignant
train_set={2:[],4:[]}
test_set={2:[],4:[]}
# just split the full data into train and test sets
train_data=full_data[:-int(test_size*len(full_data))]
test_data=full_data[-int(test_size*len(full_data)):]

# after this for loop, train_set will be a dict with 2 key-value pairs in it, one is the benign and one is the malignant group
for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct=0
total=0

for group in test_set:
    for data in test_set[group]:
        # data is the list of features of one data point
        # Here we are using our train set as the background dots, and we are trying to classify the data points within
        # the test_set into the groups within the train_set
        # The classification result is of course the vote
        # this use of train and test sets is kind of different from the traditional ML point of view
        # but this is just how k-nearest neighbors is
        vote,confidance=k_nearest_neighbors(train_set,data,k=5)
        # we know which group that each of our test data point comes from
        # if our vote(classification result) == the real group that the test point comes from
        if vote==group:
            correct +=1
        else:
            print("The wrong predictions are with confidance of: ",confidance)
        total+=1

print("Classification Accuracy:",correct/total)

# After comparing the one we build and the one in SKlearn, we can see that the accuracy of the two are similar
# but the one by sklearn is much faster, the main reason is that the one in sklearn only consider data points within
# a certain radius to the point of interest by default(only calculate the distance between the point of interest and data points
# that are "near" to that point), and of course you can use many threads if you set n_jobs to larger than 1 when you
# call the sklearn k nearest neighbor to run faster.(the default n_jobs is 1)