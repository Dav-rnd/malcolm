from typing import Dict
from h2o.estimators.gbm import H2OGradientBoostingEstimator

from models.h2o.h2o_model import H2OModel


class H2OGradientBoosting(H2OModel):
    """
    Gradient Boosting with H2O
    http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2ogradientboostingestimator
    """

    def _train(self):
        self.model = H2OGradientBoostingEstimator(**self.hyperparameters)
        self.model.train(x=self.features, y=self.target,
                         training_frame=self.training_h2o, validation_frame=self.validation_h2o)
