import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    make_union(VotingClassifier([("est", LogisticRegression(C=41.0, dual=False, penalty="l1"))]), FunctionTransformer(lambda X: X)),
    make_union(VotingClassifier([("est", MultinomialNB(alpha=0.21, fit_prior=True))]), FunctionTransformer(lambda X: X)),
    BernoulliNB(alpha=5.0, binarize=0.17, fit_prior=True)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
