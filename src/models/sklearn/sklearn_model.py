# *********************************************************
# TRAINING/TESTING SCRIPT
# (C) David Renaudie 2019
# *********************************************************
import os
import abc
import logging
from typing import Dict, List
import numpy as np
import pickle

from models.model import Model
from load_dataset import load_preproc_dataset


class SklearnModel(Model):
    MODEL_FILENAME = 'model.pkl'

    def __init__(self, dataset_name: str,
                 hyperparameters: Dict, infra_s3: Dict, features: List, target: str,
                 data_dir: str, training_job_dir: str, clean: bool=False, model_id: str=None):
        Model.__init__(self, dataset_name, hyperparameters, infra_s3, features, target,
                       data_dir, training_job_dir, clean)
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

    @abc.abstractmethod
    def _predict(self):
        raise NotImplementedError('_predict method not implemented')

    def train(self):
        logging.info('======== TRAINING ========')
        logging.info('Loading training data...')
        self.load_training_data()

        logging.info('Fitting model...')
        self._train()

        self.save()

    def predict(self):
        logging.info('======= PREDICTION =======')
        if not self.model:
            self.load()
        if self.training_x is None:
            self.load_training_data()
        self.load_testing_data()

        self._predict()

    def evaluate(self):
        if not self.model:
            logging.info('Executing prediction step to perform evaluation')
            self.predict()
        super(SklearnModel, self).evaluate()

    def load(self):
        """ Load a saved model from the disk """
        logging.info('Loading model from {}...'.format(self.model_filename))
        with open(self.model_filename, 'rb') as file:
            self.model = pickle.load(file)

    def save(self):
        """ Save the trained model to disk """
        logging.info('Saving model to {}...'.format(self.model_filename))
        pickle.dump(self.model, open(self.model_filename, 'wb'))

    def prepare_evaluation_data(self):
        self.load_training_data()
        self.load_testing_data()

    def load_training_data(self):
        logging.info('Loading training/validation datasets...')
        postfixed_dataset_name = self.dataset_name + '_preprocessed'
        s3_bucket = None if self.infra_s3 is None else self.infra_s3['s3_bucket']
        s3_folder = None if self.infra_s3 is None else self.infra_s3['s3_folder_data']
        training, validation, _, _ = load_preproc_dataset(postfixed_dataset_name, self.data_dir,
                                                          s3_bucket, s3_folder,
                                                          self.clean, True, self.target)
        self._init_features(training)
        self.training = training
        self.validation = validation

        self.training_x = self.training[self.features].values[:]
        self.validation_x = self.validation[self.features].values[:]

        self.training_y = np.ravel(self.training[self.target])
        self.validation_y = np.ravel(self.validation[self.target])
        self.n_classes = len(set(self.training_y))

        logging.info('training_x shape: {}'.format(self.training_x.shape))
        logging.info('training_y shape: {}'.format(self.training_y.shape))
        logging.info('validation_x shape: {}'.format(self.validation_x.shape))
        logging.info('validation_y shape: {}'.format(self.validation_y.shape))

    def load_testing_data(self):
        logging.info('Loading testing dataset...')
        postfixed_dataset_name = self.dataset_name + '_preprocessed'
        s3_bucket = None if self.infra_s3 is None else self.infra_s3['s3_bucket']
        s3_folder = None if self.infra_s3 is None else self.infra_s3['s3_folder_data']
        _, _, testing, _ = load_preproc_dataset(postfixed_dataset_name, self.data_dir,
                                                s3_bucket, s3_folder,
                                                self.clean, True, self.target)
        self._init_features(testing)
        self.testing = testing
        self.testing_x = self.testing[self.features].values[:]
        self.testing_y = np.ravel(self.testing[self.target])

        logging.info('testing_x shape: {}'.format(self.testing_x.shape))
        logging.info('testing_y shape: {}'.format(self.testing_y.shape))
