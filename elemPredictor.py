# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 19:24:56 2017

@author: smith
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from tpot import TPOTClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import ElasticNet
from sklearn import preprocessing, cross_validation, neighbors, ensemble,tree, svm
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.datasets import make_regression
from chemspipy import ChemSpider
import pandas as pd
import numpy as np

cs = ChemSpider('840a4e86-df1e-4c91-ba2c-d0918ffd21cb')

#print pd.read_csv('Book2.csv')
cl='halfCat'
csvInate=""
df=pd.DataFrame(pd.read_csv('data7.csv'))
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

c1 = cs.get_compound(236)  # Specify compound by ChemSpider ID
c2 = cs.search('benzene')  # Search using name, SMILES, InChI, InChIKey, etc.
X= [["protons","neutrons","electrons","periodic x","periodic y", ],["protons","ect"]]
#X=[[1,3,5,7],[11,13,15,17],[21,23,25,27],[31,33,35,37],[41,43,45,47]]
#X=[[51,53,55,57],[61,63,65,67]]
#X = [[0, 0], [1, 1], [2, 2], [3, 3],[5,3],[7,2],[6,9],[2,6],[40,68]]
#actual
#X=[[1,0,1,1,1],[2,2,2,18,1],[3,4,3,1,2],[4,5,4,2,2],[5,6,5,13,2],[6,6,6,14,2],[7,7,7,15,2],[8,8,8,16,2],[9,10,9,17,2],[37,48,37,1,5],[115,174,115,15,7],[99,153,99,14,7],[73,108,73,5,6],[25,30,25,7,4],[117,175,117,17,7],[86,136,86,18,6],[36,48,36,18,3],[30,35,30,12,4],[112,173,112,12,7],[110,171,110,10,7]]
#X= np.array(df.drop(['half'],1))
#y=np.array(df['half'])
X= np.array(df.drop([cl],1))
y=np.array(df[cl])

def rep():
    d=0
    ac=0
    n=100
    csvInate=""
    for i in range(0,n):
        d+=1
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.5)

        #clf=svm.SVM()
       # clf = neighbors.KNeighborsClassifier()
        #clf = MLPClassifier()
        #clf = ElasticNet()
        clf.fit(X_train,y_train)

        accuracy = clf.score(X_test,y_test)
        ac+=accuracy
        csvInate=csvInate+ str(accuracy)+","
    ac=ac/n
    print("acc2")
    print(ac)
    print(d)
    #print(csvInate)
        
#X =[[1,0,1,1,1,1,1,1],[2,2,2,2,2,2,18,1],[3,4,3,11,12,11,1,2],[4,5,4,12,121,12,2,2],[5,6,5,121,122,121,13,2],[6,6,6,222,222,222,14,2],[7,7,7,223,223,223,15,2],[8,8,8,224,224,224,16,2],[9,10,9,225,226,225,17,2],[37,48,37,41,4210,41,1,5],[115,174,115,6214103,6214103218141060,6214103,15,7],[99,153,99,6211,621410321811,6211,14,7],[73,108,73,62143,62146,62143,5,6],[25,30,25,325,3210,325,7,4],[117,175,117,6214105,6214103218141060,6214105,17,7],[86,136,86,6214106,6214103212,6214106,18,6],[36,48,36,42106,4210,42106,18,3],[30,35,30,4210,42105,4210,12,4],[112,173,112,621410,6214103218141060,621410,12,7],[110,171,110,62148,62141031814105,62148,10,7],[6,6,6,122,122,122,14,2],[14,14,14,222,222,222,14,3],[83,126,83,214103,52146,214103,15,6],[75,111,75,52145,62149,52145,7,6],[11,22,11,21,322,21,1,3],[118,176,118,6214106,6214103218141060,6214106,18,7],[24,28,24,324,328,324,6,4],[48,66,48,5210,5210,5210,12,5],[1,0,1,1,1,1,1,1],[2,2,2,2,2,2,18,1],[3,4,3,11,12,11,1,2],[4,5,4,12,121,12,2,2],[5,6,5,121,122,121,13,2],[6,6,6,222,222,222,14,2],[7,7,7,223,223,223,15,2],[8,8,8,224,224,224,16,2],[9,10,9,225,226,225,17,2],[37,48,37,41,4210,41,1,5],[115,174,115,6214103,6214103218141060,6214103,15,7],[99,153,99,6211,621410321811,6211,14,7],[73,108,73,62143,62146,62143,5,6],[25,30,25,325,3210,325,7,4],[117,175,117,6214105,6214103218141060,6214105,17,7],[86,136,86,6214106,6214103212,6214106,18,6],[36,48,36,42106,4210,42106,18,3],[30,35,30,4210,42105,4210,12,4],[112,173,112,621410,6214103218141060,621410,12,7],[110,171,110,62148,62141031814105,62148,10,7],[6,6,6,122,122,122,14,2],[14,14,14,222,222,222,14,3],[83,126,83,214103,52146,214103,15,6],[75,111,75,52145,62149,52145,7,6],[11,22,11,21,322,21,1,3],[118,176,118,6214106,6214103218141060,6214106,18,7],[24,28,24,324,328,324,6,4],[48,66,48,5210,5210,5210,12,5]]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.3)
#clf = neighbors.KNeighborsClassifier()
#clf=ensemble.RandomForestClassifier()
clf = TPOTClassifier(generations=50,population_size=100, verbosity=2,scoring="accuracy")
#clf = ElasticNet()
#clf = MLPClassifier()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print("acc")
print(accuracy)

#y=[[9,19,29,39,49],[9,19,29,39,49]]
#y = [0, 2, 4, 6, 8, 9,15,8,108]
#actual
#y =[143191600000000000,143191600000000000,143191600000000000,143191600000000000,143191600000000000,143191600000000000,143191600000000000,143191600000000000,143191600000000000,48400000000,0.22,40754880,143191600000000000,143191600000000000,0.8,330350.4,143191600000000000,143191600000000000,30,14,143191600000000000,143191600000000000,143191600000000000,41200000000,143191600000000000,0.008,143191600000000000,143191600000000000]
#y =[143191600000000000,143191600000000000,143191600000000000,143191600000000000,143191600000000000,143191600000000000,143191600000000000,143191600000000000,143191600000000000,48400000000,0.22,40754880,143191600000000000,143191600000000000,0.8,330350.4,143191600000000000,143191600000000000,30,14,143191600000000000,143191600000000000,143191600000000000,41200000000,143191600000000000,0.008,143191600000000000,143191600000000000,143191600000000000,143191600000000000,143191600000000000,143191600000000000,143191600000000000,143191600000000000,143191600000000000,143191600000000000,143191600000000000,48400000000,0.22,40754880,143191600000000000,143191600000000000,0.8,330350.4,143191600000000000,143191600000000000,30,14,143191600000000000,143191600000000000,143191600000000000,41200000000,143191600000000000,0.008,143191600000000000,143191600000000000]

#X, y = make_regression(n_features=5, random_state=0)
#regr = ElasticNet(random_state=0);
#regr =ElasticNet()
#regr.fit(X, y)

#ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,
      #max_iter=1000, normalize=False, positive=False, precompute=False,
      #random_state=0, selection='cyclic', tol=0.0001, warm_start=False)

#regr.fit(X, y)840a4e86-df1e-4c91-ba2c-d0918ffd21cb
#print(c2)
#print(c1)
#print(c1.common_name)
c = cs.get_compound(2157)
print(c.molecular_formula)

def exFunc():
    print("called")
    return;
         
exFunc()
#rep()
clf.export('pipeline30.py')
print("EXPORTED")



#Best pipeline: AdaBoostClassifier(LinearSVC(input_matrix, 0.25, 100, False), 0.88)

#pred_measures=np.array([[106,166,106]])
#pred_measures=np.array([[100,157,100]])
#prediction= clf.predict(pred_measures)
#print(prediction)
#NOTE: tends to estimate half lives shorter than they are

