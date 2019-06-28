import os
import abc
import logging
import numpy as np
from typing import Dict
from collections import Counter

from metrics import log_metrics
from preprocessors.preprocessing import Preprocessor


class Model(object, metaclass=abc.ABCMeta):
    def __init__(self, dataset_name: str, hyperparameters: Dict, infra_s3: Dict, features: list, target: str,
                 data_dir: str, training_job_dir: str=None, clean: bool=False):
        self.dataset_name = dataset_name
        self.hyperparameters = hyperparameters
        self.infra_s3 = infra_s3
        self.features = features
        self.target = target
        self.data_dir = data_dir
        self.training_job_dir = training_job_dir
        self.init_features = True

        self.model = None

        self.training = None
        self.validation = None
        self.testing = None

        self.training_x = None
        self.validation_x = None
        self.testing_x = None

        self.training_y = None
        self.validation_y = None
        self.testing_y = None

        self.training_y_pred = None
        self.validation_y_pred = None
        self.testing_y_pred = None

        self.n_classes = None
        self.clean = clean

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError('Need to define a train method')

    @abc.abstractmethod
    def predict(self):
        raise NotImplementedError('Need to define a predict method')

    @abc.abstractmethod
    def prepare_evaluation_data(self):
        raise NotImplementedError('Need customize the method to format data to evaluate model')

    def grid_search(self):
        logging.error('Grid_search method not implemented')

    def deploy(self):
        logging.error('Deploy method not implemented')

    def delete_endpoint(self):
        logging.error('Delete_endpoint method not implemented')

    def evaluate(self):
        logging.info('======= EVALUATION =======')
        logging.info('Starting evaluation...')
        if (self.training_y_pred is None or self.validation_y_pred is None
                or self.testing_y_pred is None):
            try:
                self.prepare_evaluation_data()
            except FileNotFoundError:
                logging.error('no predictions data nor files, cannot compute metrics. Exiting.')
                raise

        logging.info('Results on Training dataset:')
        self._log_class_distribution(self.training_y)
        log_metrics(self.training_y, self.training_y_pred, self.n_classes)

        logging.info('Results on Validation dataset:')
        self._log_class_distribution(self.validation_y)
        threshold = log_metrics(self.validation_y, self.validation_y_pred, self.n_classes)

        logging.info('Results on Testing dataset:')
        self._log_class_distribution(self.testing_y)
        log_metrics(self.testing_y, self.testing_y_pred, self.n_classes, threshold=threshold,
                    plot=True, folder_path=self.training_job_dir)

    def _init_features(self, df):
        if not self.init_features:
            return

        if not self.features:
            self.features = list(df.columns)
        else:
            # Features to use specified by the user
            # SageMaker and AWS configs may specify to use categorical features, though we added the encoded
            # value as a suffix + '$' to their original name, so mismatch to fix
            ohe_features = [f for f in df.columns if Preprocessor.OHE_NAME_SEP in f]
            ohe_original_names = list(set(map(lambda c: c.split(Preprocessor.OHE_NAME_SEP)[0], ohe_features)))
            # OHE or categ features have already been removed from DF depending on training method
            # If OHE features are still present in the DF, they won't match the specified list of features, so we add them
            if ohe_features:
                # OHE features to keep
                specified_ohe_features = [f for f in ohe_features if f.split(Preprocessor.OHE_NAME_SEP)[0] in self.features]
                # OHE original feature names removed
                self.features = [f for f in self.features if f not in ohe_original_names]
                # OHE new names added
                self.features = specified_ohe_features + self.features

        if self.target:
            if self.target not in df.columns:
                raise ValueError('Cannot find target field \'{}\' in the dataframe. Available features are {}'.format(
                                 self.target, self.features))

            if self.target in self.features:
                self.features.remove(self.target)

        logging.info('Final features: {}'.format(', '.join(self.features)))

        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            raise ValueError('Features from the config file are missing in the dataset: {}'.format(missing_features))

        self.init_features = False

    def _log_class_distribution(self, y_true):
        counts = Counter(y_true).most_common()
        logging.info('Total class distribution (ID, #): {}'.format(counts))
        logging.info('Total class proportions (ID, %): {}'.format([(cid, np.round(cnt / len(y_true) * 100, 2)) for (cid, cnt) in counts]))
