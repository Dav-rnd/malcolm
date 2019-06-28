# *********************************************************
# AWS LINEAR LEARNER SCRIPT
# (C) Remi Domingues 2019
# *********************************************************
import os
import logging
import sagemaker
import pandas as pd
from typing import Dict, List
from sagemaker.amazon.amazon_estimator import get_image_uri

from models.aws.aws_model import AWSModel
from save_datasets import upload_file_to_s3


class AWSLinear(AWSModel):
    """
    Linear learner with AWS
    https://docs.aws.amazon.com/sagemaker/latest/dg/ll_hyperparameters.html
    """

    def __init__(self, dataset_name: str, hyperparameters: Dict, infra_s3: Dict, infra_sm: Dict,
                 features: List, target: str, data_dir: str, training_job_common_dir: str, training_job_dir: str,
                 aws_model_id: str=None, clean: bool=False):
        AWSModel.__init__(self, dataset_name, hyperparameters, infra_s3, infra_sm, features, target,
                          data_dir, training_job_common_dir, training_job_dir, aws_model_id, clean)

        self.csv_training_full_filename = 'training_full.csv'
        self.csv_validation_full_filename = 'validation_full.csv'

        self.s3_training_full_csv_path = self.s3_filepath(filetype='ml_data', filename=self.csv_training_full_filename, common=True, uri=True)
        self.s3_validation_full_csv_path = self.s3_filepath(filetype='ml_data', filename=self.csv_validation_full_filename, common=True, uri=True)

    def _get_container(self, boto3_session):
        """ Return the URI corresponding to the container of the algorithm """
        return get_image_uri(boto3_session.region_name, 'linear-learner')

    def _init_s3_train_files(self):
        """ Initialize the training and validation files (features + label) required for the training step """
        # LinearModel requires CSV training and validation files, including labels, when invoking fit()
        logging.info('Preparing csv training data...')
        self._prepare_csv_full_file(self.training, self.csv_training_full_filename)
        self._prepare_csv_full_file(self.validation, self.csv_validation_full_filename)

        self.hyperparameters['feature_dim'] = self.training_x.shape[1]
        self.hyperparameters['num_classes'] = self.n_classes
        # binary_classifier, multiclass_classifier, or regressor
        if 'predictor_type' not in self.hyperparameters:
            self.hyperparameters['predictor_type'] = 'binary_classifier' if self.n_classes == 2 else 'multiclass_classifier'

        s3_input_training = sagemaker.s3_input(s3_data=self.s3_training_full_csv_path, content_type='text/csv')
        s3_input_validation = sagemaker.s3_input(s3_data=self.s3_validation_full_csv_path, content_type='text/csv')
        return s3_input_training, s3_input_validation

    def _parse_preds_line(self, preds_line):
        """ Parse the given line in order to return an array of n_classes probabilities """
        # The output for LinearModel for multiclass (3 here) and for each sample is
        # {"predicted_label":1.0,"score":[0.191362738609313,0.516198694705963,0.2924385368824]}
        return eval(preds_line)['score']

    def _prepare_csv_full_file(self, dataset: pd.DataFrame, filename) -> str:
        """ Generate the local and S3 csv files used for the training
        Similar to prepare_csv_file, but also exports the label in the first column, instead of excluding it """
        logging.info('Preparing csv full file: {}'.format(filename))
        s3_csv_file = self.s3_filepath(filetype='ml_data', filename=filename, common=True)

        if self.clean or not self.is_s3_file(s3_csv_file):
            logging.info('S3 csv file does not exist')
            local_csv_file = self.local_filepath(filename, common=True)
            if self.clean or not os.path.isfile(local_csv_file):
                logging.info('Local csv file does not exist. Computing...')
                dataset[[self.target] + self.features].to_csv(local_csv_file, header=False, index=False)
            logging.info('Local csv file found. Uploading to S3...')
            upload_file_to_s3(local_csv_file, self.infra_s3['s3_bucket'], s3_csv_file)
        else:
            logging.info('S3 csv file already exists. Skipping step.')

        logging.info('S3 csv file path: {}'.format(s3_csv_file))
