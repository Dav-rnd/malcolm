# *********************************************************
# ALTERNATIVE-LEVEL TRAINING/TESTING SCRIPT
# (C) David Renaudie 2019
# (C) Remi Domingues 2019
# *********************************************************

# =========================================================
# IMPORTS
# =========================================================
import argparse
import logging.config
import os
import time
from datetime import datetime

from models.aws import *
from models.h2o import *
from models.sklearn import *
from utils import timing, load_config, load_infra_s3, load_infra_sm

# =========================================================
# GLOBALS
# =========================================================
logging.config.fileConfig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../config/logging.conf'))


# =========================================================
# FUNCTIONS
# =========================================================

def check_args(args):
    if 't' in args.actions and args.model_id:
        raise ValueError('Parameter conflict: cannot enable both \'-a t\' and \'--model-id\'')

    if 'd' in args.actions and not args.model_id and 't' not in args.actions:
        raise ValueError('Deployment requested (-a d), but missing model training (-a t) or model ID (--model-id)')


def train_level_model(train_args) -> None:

    # timing
    start_time = time.time()
    latest_time = start_time

    # destination folder and files
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]
    training_job_common_dir = os.path.join(train_args.working_folder, train_args.dataset_name,
                                           train_args.config_id)
    training_job_dir = os.path.join(training_job_common_dir, timestamp)

    if not os.path.exists(training_job_dir):
        logging.info('creating training job dir: {}'.format(training_job_dir))
        os.makedirs(training_job_dir)

    # set logging
    fh = logging.FileHandler(os.path.join(training_job_dir, 'logging.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d: %(message)s')
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)

    # loading model parameters and performing feature selection
    logging.info('Loading training parameters...')
    logging.info('Training config file: {}'.format(train_args.config_filename))
    logging.info('Training model id: {}'.format(train_args.config_id))

    config = load_config('model_{}.yaml'.format(train_args.config_filename))
    features, target, algo, hyperparameters = get_training_config(config, train_args.config_id)
    infra_s3 = load_infra_s3(args.infra_s3)
    infra_sm = load_infra_sm(args.infra_sm)
    logging.info('Infra S3: {}'.format(infra_s3))
    logging.info('Infra SageMaker: {}'.format(infra_sm))

    logging.info('Selected features: {}'.format(features if features else 'all'))
    logging.info('Target: {}'.format(target))
    if train_args.model_id:
        logging.info('AWS model ID: {}'.format(train_args.model_id))

    # initializing model object
    logging.info('Initializing model object...')

    if 'sklearn' in algo.lower():
        model = eval(algo)(train_args.dataset_name, hyperparameters, infra_s3, features, target,
                           train_args.data_folder, training_job_dir, train_args.clean, train_args.model_id)
    elif 'aws' in algo.lower():
        if infra_s3 is None or infra_sm is None:
            raise ValueError('Parameters --infra-s3 and --infra-sm are required for SageMaker algorithms')
        model = eval(algo)(train_args.dataset_name, hyperparameters, infra_s3, infra_sm, features, target,
                           train_args.data_folder, training_job_common_dir, training_job_dir, train_args.model_id,
                           train_args.clean)
    elif 'h2o' in algo.lower():
        model = eval(algo)(train_args.dataset_name, hyperparameters, infra_s3, features, target, train_args.h2o,
                           train_args.data_folder, training_job_dir, train_args.clean, train_args.model_id)
    else:
        logging.error('Unknown algo: {}. Exiting.'.format(algo))
        raise ValueError

    # Training
    if 't' in train_args.actions:
        logging.info('training {}...'.format(algo))
        logging.info('hyper-parameters: {}'.format(hyperparameters))
        model.train()
        timing('training', latest_time)

    # Predict
    if 'p' in train_args.actions:
        model.predict()
        timing('predict', latest_time)

    # Evaluate results
    if 'e' in train_args.actions:
        model.evaluate()

    # Deploy
    if 'd' in train_args.actions:
        model.deploy()

    # Remove
    if 'r' in train_args.actions:
        model.delete_endpoint()

    # Grid search
    if 'g' in train_args.actions:
        logging.info('grid search {}...'.format(algo))
        model.grid_search()
        timing('grid search', latest_time)

    # Final timing
    timing('total time', start_time)

    return


def get_training_config(config, config_id):
    """
    :param config: input loaded config
    :param config_id: id of the model to use
    :return: endpoint_name, features for this model
    """
    if config_id not in config:
        raise KeyError('Training config id "{}" not found in config file. Exiting.'.format(config_id))
    config_model_info = config[config_id]

    try:
        features = config_model_info['features']
        target = config_model_info['target']
        algo = config_model_info['algo']
        hyperparameters = config_model_info['hyperparameters']
    except KeyError:
        logging.error('Config file formatting: information not found for requested model_id. Exiting.')
        raise

    return features, target, algo, hyperparameters


# =========================================================
# MAIN
# =========================================================
if __name__ == '__main__':
    # Input parsing
    parser = argparse.ArgumentParser(description='TRAINING/TESTING/DEPLOYMENT SCRIPT ')
    parser.add_argument('-a', dest='actions', type=str, help='actions to perform among: t(training), p(predict), e(evaluate), g(grid search),\n\
                                                             d(deploy endpoint): deploy the model trained or specified by --model-id on SageMaker,\n\
                                                             r(remove endpoint): delete the endpoint corresponding to the specified --model-id', required=True)
    parser.add_argument('-d', dest='dataset_name', type=str, help='input dataset name (without extension)', required=True)
    parser.add_argument('-f', dest='data_folder', type=str, help='data folder (default data)', default='data')
    parser.add_argument('-w', dest='working_folder', type=str, help='working folder (default jobs)', default='jobs')
    parser.add_argument('-c', dest='config_filename', type=str, required=True,
                        help='config file name based on `config/model_<configfilename>.yaml`')
    parser.add_argument('--model', dest='config_id', type=str, required=True,
                        help='model config with a matching in `config/model_<configfilename>.yaml`')
    parser.add_argument('--h2o', dest='h2o', type=str, default='localhost:54321', help='ip:port of a running h2o instance')
    parser.add_argument('--infra-s3', dest='infra_s3', type=str, default=None, help='configuration of the S3 bucket from infra_s3.yaml (required for AWS algos)')
    parser.add_argument('--infra-sm', dest='infra_sm', type=str, default=None, help='configuration of the SageMaker platform from infra_sm.yaml (required for AWS algos)')
    parser.add_argument('--model-id', dest='model_id', type=str, default=None, help='model ID of a previously trained model. Ignored if \'t\' is part of the actions')
    parser.add_argument('--clean', dest='clean', help='ignore previously generated local files and erase intermediate files to regenerate them', action='store_true')
    args = parser.parse_args()

    check_args(args)
    train_level_model(args)
