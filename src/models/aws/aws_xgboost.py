# *********************************************************
# AWS XGBOOST SCRIPT
# (C) David Renaudie 2019
# (C) Remi Domingues 2019
# *********************************************************
import os
import logging
import sagemaker
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.datasets import dump_svmlight_file
from sagemaker.amazon.amazon_estimator import get_image_uri

from models.aws.aws_model import AWSModel
from save_datasets import upload_file_to_s3


class AWSXGBoost(AWSModel):
    """
    eXtreme Gradient Boosting (XGBoost) with AWS
    https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html
    """

    def __init__(self, dataset_name: str, hyperparameters: Dict, infra_s3: Dict, infra_sm: Dict, features: List,
                 target: str, data_dir: str, training_job_common_dir: str, training_job_dir: str,
                 aws_model_id: str=None, clean: bool=False):
        AWSModel.__init__(self, dataset_name, hyperparameters, infra_s3, infra_sm, features, target,
                          data_dir, training_job_common_dir, training_job_dir, aws_model_id, clean)

        self.s3_training_file = self.s3_filepath(filetype='ml_data', filename='training.libsvm', common=True)
        self.s3_validation_file = self.s3_filepath(filetype='ml_data', filename='validation.libsvm', common=True)

        self.s3_training_libsvm_path = self.s3_to_uri(self.s3_training_file)
        self.s3_validation_libsvm_path = self.s3_to_uri(self.s3_validation_file)

        self.local_libsvm_training_file = self.local_filepath('training.libsvm', common=True)
        self.local_libsvm_validation_file = self.local_filepath('validation.libsvm', common=True)

    def _get_container(self, boto3_session):
        """ Return the URI corresponding to the container of the algorithm """
        return get_image_uri(boto3_session.region_name, 'xgboost')

    def _init_s3_train_files(self):
        """ Initialize the training and validation files (features + label) required for the training step """
        # XGBoost requires libsvm training and validation files when invoking fit()
        self._prepare_libsvm_data()

        if self.n_classes <= 2:
            self.hyperparameters['eval_metric'] = 'auc'
            self.hyperparameters['objective'] = 'binary:logistic'
        else:
            self.hyperparameters['objective'] = 'multi:softprob'
            self.hyperparameters['num_class'] = self.n_classes

        s3_input_training = sagemaker.s3_input(s3_data=self.s3_training_libsvm_path, content_type='libsvm')
        s3_input_validation = sagemaker.s3_input(s3_data=self.s3_validation_libsvm_path, content_type='libsvm')
        return s3_input_training, s3_input_validation

    def _parse_preds_line(self, preds_line):
        """ Parse the given line in order to return an array of n_classes probabilities """
        # The output for AWS XGBoost for multiclass is [proba_c1, probac2, probac3,...] for each sample, neither csv, nor python...
        # return list(map(float, preds_line[1:-2].split(',')))
        return eval(preds_line)

    def _prepare_libsvm_data(self):
        """ Generate the local and S3 libsvm files used for the training """
        logging.info('Preparing libsvm training data...')
        if self.clean or not (self.is_s3_file(self.s3_training_file) and self.is_s3_file(self.s3_validation_file)):
            logging.info('S3 libsvm files do not exist.')
            if self.clean or not (os.path.isfile(self.local_libsvm_training_file) and os.path.isfile(self.local_libsvm_validation_file)):
                logging.info('Local libsvm files do not exist.')
                logging.info('Generating local libsvm files...')
                dump_svmlight_file(X=self.training_x, y=self.training_y, f=self.local_libsvm_training_file)
                dump_svmlight_file(X=self.validation_x, y=self.validation_y, f=self.local_libsvm_validation_file)
            logging.info('Local libsvm files found. Uploading to S3...')
            upload_file_to_s3(self.local_libsvm_training_file, self.infra_s3['s3_bucket'], self.s3_training_file)
            upload_file_to_s3(self.local_libsvm_validation_file, self.infra_s3['s3_bucket'], self.s3_validation_file)
        else:
            logging.info('S3 libsvm files already exist. Skipping step.')
