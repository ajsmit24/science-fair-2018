import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('data8.csv', delimiter=',', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=42)
#print(testing_features)

exported_pipeline = make_pipeline(
    Nystroem(gamma=15.0, kernel="linear", n_components=30),
    AdaBoostClassifier(learning_rate=0.73, n_estimators=500)
)

exported_pipeline.fit(training_features, training_classes)
accuracy = exported_pipeline.score(testing_features,testing_classes)
print(accuracy)
results = exported_pipeline.predict(testing_features)
print(results)
def rep():
    d=0
    ac=0
    n=100
    for i in range(0,n):
        if i==n/2:
            print("50%")
        if i==n/4:
            print("25%")
        if i==((n/4)*3):
            print("75%")
        d+=1
        training_features, testing_features, training_classes, testing_classes = \
            train_test_split(features, tpot_data['class'], random_state=42)

        exported_pipeline = make_pipeline(
            Nystroem(gamma=15.0, kernel="linear", n_components=30),
            AdaBoostClassifier(learning_rate=0.73, n_estimators=500)
        )
        exported_pipeline.fit(training_features, training_classes)
        accuracy = exported_pipeline.score(testing_features,testing_classes)
        ac+=accuracy
    ac=ac/n
    print("ac AV")
    print(ac)
rep()
#print(exported_pipeline.predict([[56,74,56,2,6,0,0]]))
#69.87% 
