# *********************************************************
# AWS LINEAR LEARNER SCRIPT
# (C) Remi Domingues 2019
# *********************************************************
import sagemaker
from typing import Dict, List
from sagemaker.sklearn.estimator import SKLearn

from models.aws.aws_model import AWSModel


class AWSSklearn(AWSModel):
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

    def _get_estimator(self, sm_session):
        # TODO: hyperparameters must be transformed into a string
        # TODO: import must be passed from config file!
        # proposal: boolean to deploy for sklearn models + additional import field if deployed
        s_import = 'from sklearn.ensemble import RandomForestClassifier'
        hyperparameters = '{"n_estimators": 1000, "max_depth": 10}'

        return SKLearn(entry_point='src/models/aws/train_sklearn.py',
                       role=self.infra_sm['sm_role'],
                       train_instance_count=self.infra_sm['train_instance_count'],
                       train_instance_type=self.infra_sm['train_instance_type'],
                       output_path=self.s3_filepath(filetype='model', filename='', uri=True),
                       sagemaker_session=sm_session, train_max_run=self.infra_sm['maxruntime'],
                       hyperparameters={'import': s_import,
                                        'hyperparameters': hyperparameters})

    def _get_transformer_class(self):
        return sagemaker.transformer.Transformer

    def _get_container(self, boto3_session):
        """ Return the URI corresponding to the container of the algorithm """
        return None

    def _init_s3_train_files(self):
        """ Initialize the training and validation files (features + label) required for the training step """
        # LinearModel requires CSV training and validation files, including labels, when invoking fit()
        return self._init_s3_train_csv_files()

    def _parse_preds_line(self, preds_line):
        """ Parse the given line in order to return an array of n_classes probabilities """
        # The output for LinearModel for multiclass (3 here) and for each sample is
        # {"predicted_label":1.0,"score":[0.191362738609313,0.516198694705963,0.2924385368824]}
        # return eval(preds_line)['score']
        print(preds_line)
        return None
