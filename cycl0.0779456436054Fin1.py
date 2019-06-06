import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    SelectFwe(alpha=0.05, score_func=f_classif),
    make_union(VotingClassifier([("est", AdaBoostClassifier(learning_rate=0.92, n_estimators=500))]), FunctionTransformer(lambda X: X)),
    make_union(VotingClassifier([("est", BernoulliNB(alpha=0.26, binarize=13.0, fit_prior=True))]), FunctionTransformer(lambda X: X)),
    LinearSVC(C=13.0, dual=False, penalty="l1")
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
