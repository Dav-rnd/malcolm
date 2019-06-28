from typing import Dict, List
from sklearn.svm import SVC

from models.sklearn.sklearn_model import SklearnModel


class SklearnSVM(SklearnModel):
    """
    SVM with RBF kernel
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """

    def _train(self):
        self.model = SVC(**self.hyperparameters)
        self.model.fit(self.training_x, self.training_y)

    def _predict(self):
        self.training_y_pred = self.model.predict_proba(self.training_x)
        self.validation_y_pred = self.model.predict_proba(self.validation_x)
        self.testing_y_pred = self.model.predict_proba(self.testing_x)
