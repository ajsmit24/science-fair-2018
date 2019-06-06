import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    make_union(VotingClassifier([("est", LinearSVC(C=25.0, dual=True, penalty="l2"))]), FunctionTransformer(lambda X: X)),
    Nystroem(gamma=0.46, kernel="chi2", n_components=32),
    AdaBoostClassifier(learning_rate=1.0, n_estimators=500)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
