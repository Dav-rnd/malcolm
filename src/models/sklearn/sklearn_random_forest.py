import os
import logging
from typing import Dict, List
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from models.sklearn.sklearn_model import SklearnModel


class SklearnRandomForest(SklearnModel):
    """
    Random Forest with sklearn
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    def _train(self):
        self.model = RandomForestClassifier(**self.hyperparameters)
        self.model.fit(self.training_x, self.training_y)

    def _predict(self):
        self.training_y_pred = self.model.predict_log_proba(self.training_x)
        self.validation_y_pred = self.model.predict_log_proba(self.validation_x)
        self.testing_y_pred = self.model.predict_log_proba(self.testing_x)

    def evaluate(self):
        super(SklearnRandomForest, self).evaluate()
        self.plot_feature_importance(15)

    def plot_feature_importance(self, top_k):
        features_importance_ordered = sorted(
            zip(map(lambda x: round(x, 5), self.model.feature_importances_), self.features),
            reverse=True)
        top_k_features = list(map(lambda x: x[1], features_importance_ordered))[:top_k]
        top_k_importance = list(map(lambda x: x[0], features_importance_ordered))[:top_k]
        fig, axes = plt.subplots(figsize=(16, 8), nrows=1, ncols=1)
        sns_plot = sns.barplot(y=top_k_features, x=top_k_importance, orient='h', ax=axes)
        sns_plot.set_xlabel('Feature importance')
        sns_plot.set_title('SK Learn RF variable importance')
        plt.savefig(os.path.join(self.training_job_dir, 'varimp.png'), bbox_inches='tight')
        logging.info('SK Learn RF scaled feature importance: {}'.format(repr(list(zip(top_k_features,
                                                                                      top_k_importance)))))
