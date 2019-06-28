import logging
from typing import Dict
from collections import Counter

import os
import abc
from os import listdir
import h2o
import numpy as np
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

from models.model import Model
from load_dataset import load_preproc_dataset


class H2OModel(Model):
    MODEL_FILENAME = 'model'

    def __init__(self, dataset_name: str, hyperparameters: Dict, infra_s3: Dict, features: list, target: str,
                 h2o_ip: str, data_dir: str, training_job_dir: str=None, clean: bool=False, model_id: str=None):
        Model.__init__(self, dataset_name, hyperparameters, infra_s3, features, target,
                       data_dir, training_job_dir, clean)

        self.ip = h2o_ip.split(':')[0]
        self.port = h2o_ip.split(':')[1]

        if model_id:
            self.model_id = model_id
            timestamp = self.model_id[-23:]
            self.model_filename = os.path.join(*(training_job_dir.split('/')[:-1] + [timestamp, self.MODEL_FILENAME]))
        else:
            self.model_id = '-'.join(training_job_dir.split('/')[1:])
            self.model_filename = os.path.join(training_job_dir, self.MODEL_FILENAME)

        logging.info('======== Model ID ========\n{}'.format(self.model_id))

    @abc.abstractmethod
    def _train(self):
        raise NotImplementedError('_train method not implemented')

    def train(self):
        logging.info('======== TRAINING ========')
        self.prepare_training_data()

        self._train()

        self.save()

    def predict(self):
        logging.info('======= PREDICTION =======')
        if self.training_x is None:
            self.prepare_training_data()
        if not self.model:
            self.load()

        # TODO: predict returns a H2O frame with these columns:
        # [predict, proba_c1, proba_c2, ..., proba_cal_c1, proba_cal_c2, ...]
        # where predict is the predicted label, and the other columns contain classification probability per class
        # The columns prefixed by 'cal_' are calibrated class probabilities using Platt Scaling. They may be better
        self.training_y_pred = self.model.predict(self.training_h2o).as_data_frame()
        dropped_cols = ['predict']
        if 'calibrate_model' in self.hyperparameters and self.hyperparameters['calibrate_model']:
            dropped_cols.extend([c for c in self.training_y_pred.columns if c[:4] != 'cal_' and c != 'predict'])
        self.training_y_pred = self.training_y_pred.drop(dropped_cols, axis=1).values

        self.validation_y_pred = self.model.predict(self.validation_h2o).as_data_frame().drop(dropped_cols, axis=1).values
        self.testing_y_pred = self.model.predict(self.testing_h2o).as_data_frame().drop(dropped_cols, axis=1).values

    def prepare_training_data(self):
        self.load_training_data()
        h2o.init(ip=self.ip, port=self.port)

        for feature, _type in self.feature_types.items():
            logging.info('- {}: {}'.format(feature, _type))

        # types must be in [real, enum, bool, int, numeric]
        self.training_h2o = h2o.H2OFrame(self.training, column_types=self.feature_types)
        self.validation_h2o = h2o.H2OFrame(self.validation, column_types=self.feature_types)
        self.testing_h2o = h2o.H2OFrame(self.testing, column_types=self.feature_types)

    def evaluate(self):
        if not self.model:
            logging.info('Executing prediction step to perform evaluation')
            self.predict()
        super(H2OModel, self).evaluate()
        self.feature_importance(15)

    def load(self):
        """ Load a saved model from the disk """
        logging.info('Loading model from {}...'.format(self.model_filename))
        self.model = h2o.load_model(os.path.join(self.model_filename, listdir(self.model_filename)[0]))

    def save(self):
        """ Save the trained model to disk """
        logging.info('Saving model to {}...'.format(self.model_filename))
        _ = h2o.save_model(model=self.model, path=self.model_filename, force=True)

    def feature_importance(self, top_k):
        variables = [var[0] for var in self.model.varimp()]
        scaled_importance = [var[2] for var in self.model.varimp()]

        features_importance_ordered = sorted(
            zip(map(lambda x: round(x, 5), scaled_importance), variables),
            reverse=True)
        top_k_features = list(map(lambda x: x[1], features_importance_ordered))[:top_k]
        top_k_importance = list(map(lambda x: x[0], features_importance_ordered))[:top_k]
        fig, axes = plt.subplots(figsize=(16, 8), nrows=1, ncols=1)
        sns_plot = sns.barplot(y=top_k_features, x=top_k_importance, orient='h', ax=axes)
        sns_plot.set_xlabel('Scaled importance')
        sns_plot.set_title('H2O RF variable importance')
        plt.savefig(os.path.join(self.training_job_dir, 'varimp.png'), bbox_inches='tight')
        logging.info('H2O RF scaled feature importance: {}'.format(repr(list(zip(top_k_features, top_k_importance)))))

    def load_training_data(self):
        # if self.training is not None and self.validation is not None and self.testing is not None:
        #     return

        logging.info('Loading preprocessed datasets...')
        postfixed_dataset_name = self.dataset_name + '_preprocessed'
        s3_bucket = None if self.infra_s3 is None else self.infra_s3['s3_bucket']
        s3_folder = None if self.infra_s3 is None else self.infra_s3['s3_folder_data']
        training, validation, testing, self.feature_types = load_preproc_dataset(postfixed_dataset_name, self.data_dir,
                                                            s3_bucket, s3_folder, self.clean, False, self.target)
        self._init_features(training)
        self.feature_types = {feature: _type for feature, _type in self.feature_types.items() if feature in self.features + [self.target]}
        self.training = training
        self.validation = validation
        self.testing = testing

        self.training_y = np.ravel(self.training[self.target])
        self.validation_y = np.ravel(self.validation[self.target])
        self.testing_y = np.ravel(self.testing[self.target])
        self.n_classes = len(set(self.training_y))

        logging.info('training_y shape: {}\nClass distribution: {}'.format(self.training_y.shape, Counter(self.training_y).most_common()))
        logging.info('validation_y shape: {}\nClass distribution: {}'.format(self.validation_y.shape, Counter(self.validation_y).most_common()))
        logging.info('testing_y shape: {}\nClass distribution: {}'.format(self.testing_y.shape, Counter(self.testing_y).most_common()))

    def prepare_evaluation_data(self):
        self.load_training_data()

    def grid_search(self):
        logging.info('starting grid search')

        # Random grid search with max_models specified in search_criteria
        rf_hyperparams = {
            'ntrees': list(range(50, 150, 25)),
            'mtries': [-1] + list(range(1, 6, 1)),
            'col_sample_rate_per_tree': [0.5 + i*0.1 for i in range(6)],
            'nbins': list(range(20, 101, 20)),
            'max_depth': list(range(10, 31, 5))
        }

        # Search criteria
        search_criteria = {'strategy': 'RandomDiscrete', 'max_models': 60}

        remaining_parameters = {}
        for key in self.hyperparameters:
            if key not in rf_hyperparams:
                remaining_parameters[key] = self.hyperparameters[key]

        logging.info('grid hyperparameters: {}'.format(rf_hyperparams))

        rf_grid = H2OGridSearch(model=H2ORandomForestEstimator,
                                grid_id='rf_grid',
                                hyper_params=rf_hyperparams,
                                search_criteria=search_criteria)  # only for random grid search

        rf_grid.train(x=self.features, y=self.target,
                      training_frame=self.training_h2o,
                      validation_frame=self.validation_h2o,
                      **remaining_parameters)

        grid_perf = rf_grid.get_grid(sort_by='auc', decreasing=True)

        logging.info('grid performance: \n{}'.format(grid_perf))
