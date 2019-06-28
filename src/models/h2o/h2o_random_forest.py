from typing import Dict
from h2o.estimators.random_forest import H2ORandomForestEstimator

from models.h2o.h2o_model import H2OModel


class H2ORandomForest(H2OModel):
    """
    Random Forest with H2O
    http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2orandomforestestimator
    """

    def _train(self):
        self.hyperparameters['binomial_double_trees'] = self.n_classes <= 2
        self.hyperparameters['calibrate_model'] = self.n_classes <= 2

        if self.hyperparameters['calibrate_model']:
            self.hyperparameters['calibration_frame'] = self.validation_h2o

        self.model = H2ORandomForestEstimator(**self.hyperparameters)
        self.model.train(x=self.features, y=self.target,
                         training_frame=self.training_h2o, validation_frame=self.validation_h2o)
