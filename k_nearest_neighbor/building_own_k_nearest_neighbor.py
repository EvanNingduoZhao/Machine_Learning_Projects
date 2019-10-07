import numpy as np
from math import sqrt
import  matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')

# so here we are modeling a binary classification problem,where k and r each represents a group
dataset={'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
# new_feature is the point of interest that we would like to classify
new_features=[5,7]

# the graph we are showing here is just a simple graphical illustration of what a dataset being passed to
# our algorithm might look like, it's not part of the actual algorithm code
for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0],ii[1],s=100,color=i)
plt.scatter(new_features[0],new_features[1])
plt.show()

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
    return vote_result

result= k_nearest_neighbors(dataset,new_features,k=3)
print(result)