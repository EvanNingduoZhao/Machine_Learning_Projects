import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
%matplotlib inline

wine = pd.read_csv('winequality-red.csv')
wine.head()
wine.info()
wine.isnull().sum()

#bins is using 3 numbers to divide numerical quality values into two ranges: bad and good
#you can also divide them into three ranges with four numbers
#labels specify the character value that you want to assign to each range
wine['quality']=pd.cut(wine['quality'],bins=[2,6.5,8],labels=['bad','good'])

#to see the different levels that the quality column contains and their order
wine['quality'].unique()

#change good and bad to categorical(R levels) numeric values eg:bad is now 0 and good is 1,
#if you have three categories then it will assign 0,1,2
label_quality=LabelEncoder()
wine['quality']=label_quality.fit_transform(wine['quality'])

wine.head(5)

#to see how many entries belong to each level
wine['quality'].value_counts()

sns.countplot(wine['quality'])

X=wine.drop('quality',axis=1)
y=wine['quality']

X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

wine.head(10)

#we want to scale the columns to make sure a specific column does not make much of an impact just because the values
# in that column have larger value compared to the other
#(Eg total sulfur dioxide generally have larger values than chlorides) we want to make all of them range from 0 to 1
sc=StandardScaler()
X_train =sc.fit_transform(X_train)
X_test =sc.transform(X_test)




#Random Forest Classifier

#construct model and train model
rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)

#let the trained model to make predictions
pred_rfc = rfc.predict(X_test)
pred_rfc

#test how good is our model
print(classification_report(y_test,pred_rfc))

#this is another way to see how good is our model
#the following matrix means as for actual bad wines, our model correctly identified 262 of them as bad
#but identified 11 of them wrongly as good wines. Then as for the actual good wines,our model correctly identified 26 of them
#as good ones but mis-identified 21 of them wrongly as bad wines
print(confusion_matrix(y_test,pred_rfc))




#SVM Classifier

clf=svm.SVC()
clf.fit(X_train,y_train)
pred_clf = clf.predict(X_test)
pred_clf

print(classification_report(y_test,pred_clf))
print(confusion_matrix(y_test,pred_clf))





#Neural Network

# we set three hidden_layers, each with 11 nodes (since we have 11 features)
# max_iter=500 means that we go through the three layers 500 times
mlpc=MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=500)

mlpc.fit(X_train,y_train)
pred_mlpc = mlpc.predict(X_test)
pred_mlpc
print(classification_report(y_test,pred_mlpc))
print(confusion_matrix(y_test,pred_mlpc))
from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test,pred_rfc)
cm





#Apply Our Best Model To predict the quality of a specific wine
Xnew = [[7.4,0.66,0.00,1.8,0.075,13.0,40.0,0.9978,3.51,0.56,9.4]]
Xnew=sc.transform(Xnew)
ynew=rfc.predict(Xnew)
ynew