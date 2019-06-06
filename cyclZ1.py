import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    make_union(VotingClassifier([("est", KNeighborsClassifier(n_neighbors=5, weights="uniform"))]), FunctionTransformer(lambda X: X)),
    VarianceThreshold(threshold=38.0),
    RandomizedPCA(iterated_power=10),
    RandomForestClassifier(n_estimators=500)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)