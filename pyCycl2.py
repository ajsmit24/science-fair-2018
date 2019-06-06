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

def cyc():
    n=5;
    s=0;
    cl="class"
    ans=""
    df=pd.DataFrame(pd.read_csv('colData.csv'))
    varArr = ["protons","nuetrons","electrons","ProtElectConfigNumb","NueConfigNumb","periodic x","periodic y"]
    for i in range(s,7):
        ans+='['+str(i)+'],'
        
        print('col'+str(i)+'.csv')
        print("data framed")
		
        if i != 7:
           df.drop([varArr[i]],1, inplace=True)
       
			
        df.drop(['element'],1, inplace=True)
        df.drop(['ProtElectConfig'],1, inplace=True)
        df.drop(['NueConfig'],1, inplace=True)
        
        
        if i==7:
           df.drop(['ProtElectConfigNumb'],1, inplace=True)
           df.drop(['NueConfigNumb'],1, inplace=True)
			
        if i==8:
			df.drop(['magicNue'],1, inplace=True)
			df.drop(['magicPro'],1, inplace=True)
		
		#itertools.combinations(iterable, r)?
		
		
		
        #df.drop(['half'],1, inplace=True)
        #df.drop(['magicNue'],1, inplace=True)
        #df.drop(['magicPro'],1, inplace=True)
        
        #df.replace(NaN)
        #print(df)
        
        

        X= np.array(df.drop([cl],1))
        y=np.array(df[cl])
        print("here")

        #print(csvInate)
            
        #X =[[1,0,1,1,1,1,1,1],[2,2,2,2,2,2,18,1],[3,4,3,11,12,11,1,2],[4,5,4,12,121,12,2,2],[5,6,5,121,122,121,13,2],[6,6,6,222,222,222,14,2],[7,7,7,223,223,223,15,2],[8,8,8,224,224,224,16,2],[9,10,9,225,226,225,17,2],[37,48,37,41,4210,41,1,5],[115,174,115,6214103,6214103218141060,6214103,15,7],[99,153,99,6211,621410321811,6211,14,7],[73,108,73,62143,62146,62143,5,6],[25,30,25,325,3210,325,7,4],[117,175,117,6214105,6214103218141060,6214105,17,7],[86,136,86,6214106,6214103212,6214106,18,6],[36,48,36,42106,4210,42106,18,3],[30,35,30,4210,42105,4210,12,4],[112,173,112,621410,6214103218141060,621410,12,7],[110,171,110,62148,62141031814105,62148,10,7],[6,6,6,122,122,122,14,2],[14,14,14,222,222,222,14,3],[83,126,83,214103,52146,214103,15,6],[75,111,75,52145,62149,52145,7,6],[11,22,11,21,322,21,1,3],[118,176,118,6214106,6214103218141060,6214106,18,7],[24,28,24,324,328,324,6,4],[48,66,48,5210,5210,5210,12,5],[1,0,1,1,1,1,1,1],[2,2,2,2,2,2,18,1],[3,4,3,11,12,11,1,2],[4,5,4,12,121,12,2,2],[5,6,5,121,122,121,13,2],[6,6,6,222,222,222,14,2],[7,7,7,223,223,223,15,2],[8,8,8,224,224,224,16,2],[9,10,9,225,226,225,17,2],[37,48,37,41,4210,41,1,5],[115,174,115,6214103,6214103218141060,6214103,15,7],[99,153,99,6211,621410321811,6211,14,7],[73,108,73,62143,62146,62143,5,6],[25,30,25,325,3210,325,7,4],[117,175,117,6214105,6214103218141060,6214105,17,7],[86,136,86,6214106,6214103212,6214106,18,6],[36,48,36,42106,4210,42106,18,3],[30,35,30,4210,42105,4210,12,4],[112,173,112,621410,6214103218141060,621410,12,7],[110,171,110,62148,62141031814105,62148,10,7],[6,6,6,122,122,122,14,2],[14,14,14,222,222,222,14,3],[83,126,83,214103,52146,214103,15,6],[75,111,75,52145,62149,52145,7,6],[11,22,11,21,322,21,1,3],[118,176,118,6214106,6214103218141060,6214106,18,7],[24,28,24,324,328,324,6,4],[48,66,48,5210,5210,5210,12,5]]

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.3)
        #clf = neighbors.KNeighborsClassifier()
        #clf=ensemble.RandomForestClassifier()
        clf = TPOTClassifier(generations=2,population_size=100, verbosity=2,scoring="accuracy")
        #clf = ElasticNet()
        #clf = MLPClassifier()
        clf.fit(X_train,y_train)
        
        accuracy = clf.score(X_test,y_test)
        print("acc"+i\
              )
        print(accuracy)  
        #clf.export('col2'+str(i)+'.py')
        print("EXPORTED")
        ans+=str(accuracy)+','
    print(ans)
cyc()