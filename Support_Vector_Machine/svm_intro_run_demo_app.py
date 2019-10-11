
import numpy as np
from sklearn import preprocessing,neighbors,svm
from sklearn.model_selection import train_test_split
import pandas as pd

df=pd.read_csv('breast-cancer-wisconsin.data')
# in the dataset all missing data is represented by a ?
# most algorithm can recognize -99999 means missing data
df.replace('?',-99999,inplace=True)
# since id doesn't have any effect on if a tumor is benign or malignent, we drop this column
df.drop(['id'],1,inplace=True)

X=np.array(df.drop(['class'],1))
y=np.array(df['class'])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# setting n_jobs=-1 will allow the algorithm to use as many threads as possible
# Then it can calculate the distances of the point of interest to many data points at the same time
clf = svm.SVC()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)

#here we are making up an example to pass to the k-nearest neighbor classifier
example_measures = np.array([4,2,1,1,1,2,3,2,1])

# since sklearn algorithms require a list of lists input
# the following line just change our input list example_measures from a list to a list of one list
# better look up what np.array.reshape actually does and how to use it
example_measures = example_measures.reshape(1,-1)


prediction = clf.predict(example_measures)
# the prediction returns a 2, which means it predits it as benign, since in the original dataset
# benign tumors are labels as 2 in the class col and malignent ones are labeled as 4
print(prediction)