import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer


from sklearn.cross_validation import train_test_split
from sklearn.decomposition import FastICA
from sklearn.ensemble import VotingClassifier
import pandas as pd 
from sklearn import preprocessing, cross_validation, neighbors, ensemble,tree, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
#tpot_data = pd.DataFrame(pd.read_csv('data7.csv'))
#features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
#training_features, testing_features, training_classes, testing_classes = \
 #   train_test_split(features, tpot_data['class'], random_state=42)
df=pd.DataFrame(pd.read_csv('cycl1.csv'))
print("data framed")
df.drop(['element'],1, inplace=True)
df.drop(['ProtElectConfig'],1, inplace=True)
df.drop(['NueConfig'],1, inplace=True)

#df.drop(['nuetrons'],1, inplace=True)
#df.drop(['periodic y'],1, inplace=True)
#df.drop(['periodic x'],1, inplace=True)
df.drop(['ProtElectConfigNumb'],1, inplace=True)
df.drop(['NueConfigNumb'],1, inplace=True)
#df.drop(['half'],1, inplace=True)
#df.drop(['magicNue'],1, inplace=True)
#df.drop(['magicPro'],1, inplace=True)
df.replace(['stable'],143191600000000000,inplace=True)
#df.replace(NaN)
#print(df)
X= np.array(df.drop(['halfCat'],1))
y=np.array(df['halfCat'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.25)


exported_pipeline = make_pipeline(
    SelectPercentile(percentile=11, score_func=f_classif),
    RandomizedPCA(iterated_power=10),
    GradientBoostingClassifier(learning_rate=1.0, max_features=1.0, n_estimators=500)
)



exported_pipeline.fit(X_train,y_train)
clf=exported_pipeline
accuracy = exported_pipeline.score(X_test,y_test)
oldAc=exported_pipeline.score(X_test,y_test)
def rep():
    d=0
    ac=0
    n=1000
    ans=""
    for i in range(0,n):
        global export_pipline
        global oldAc
        global clf
        if i==n/2:
            print("50%")
        if i==n/4:
            print("25%")
        if i==((n/4)*3):
            print("75%")
        d+=1
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.25)

        #exported_pipeline.fit(X_train,y_train)
        accuracy = exported_pipeline.score(X_test,y_test)
        if(accuracy>oldAc):
            clf=exported_pipeline
        ans=ans+str(accuracy)+","
        ac+=accuracy
    ac=ac/n
    print("ac AV")
    print(ac)
    print(" ")
    #print(ans)
rep()
hold12= 6.12*(10**12)
print(exported_pipeline.predict([[118,151,96,11,7]]))
atomNumb=100
numbIso=50
numbElem=50
nue=0
row=8
col=1
hold=0
tester="";
al="";
for p in range(0,numbElem):
	#print(p)
	atomNumb=atomNumb+1;
	if (col==2 and hold!=17):
		hold=hold+1
	elif(col==18):
		col=1
		row=row+1
	else:
		col=col+1
		
	
	
	for n in range(0,numbIso):
		#print("big"+str(n))
		nue=atomNumb+n
		tester=str(atomNumb)+','+str(nue)+','+str(atomNumb)+','+"--"+','+"--"+','+str(col)+','+str(row)+','+"--"+','+"--"
		al=al+tester+","+str(clf.predict([[atomNumb,nue,atomNumb,col,row]]))
	for n in range(0,numbIso):
		#print("little"+str(n))
		nue=atomNumb-n
		tester=str(atomNumb)+','+str(nue)+','+str(atomNumb)+','+"--"+','+"--"+','+str(col)+','+str(row)+','+"--"+','+"--"
		al=al+tester+","+str(clf.predict([[atomNumb,nue,atomNumb,col,row]]))
print(al)
 