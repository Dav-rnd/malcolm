# *********************************************************
# PREPROCESSING - PREPARE_DATASET
# (C) Alexandre Salch 2019
# (C) David Renaudie 2019
# *********************************************************

# =========================================================
# IMPORTS
# =========================================================
import logging.config
import os
import time
import argparse

from preprocessors.preprocessing import Preprocessor
from save_datasets import save_datasets
from load_dataset import load_raw_dataset
from utils import timing, load_infra_s3

logging.config.fileConfig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../config/logging.conf'))

TRAIN_PROPORTION = 0.6
VALID_PROPORTION = 0.2
TEST_PROPORTION = 0.2


def get_preprocessor_class(preprocessor_name):
    """ Return the Preprocessor<Name> object from the corresponding file in the preprocessors folder """
    if not preprocessor_name:
        return Preprocessor
    filename = 'preprocessors.preprocessing_{}'.format(preprocessor_name)
    preprocessor_class_name = 'Preprocessor{}'.format(preprocessor_name.capitalize())
    s_import = 'from {} import {}'.format(filename, preprocessor_class_name)
    exec(s_import)
    return eval(preprocessor_class_name)


def prepare_datasets(dataset_name: str, working_folder: str, s3_bucket: str=None, s3_folder: str=None,
                     target: str=None, headers: bool=False, verbose: bool=False, clean: bool=False,
                     preprocessor_name: str=None, scale: bool=False, categ_encoding: bool=False,
                     categ_features: list=None):
    # timing
    start_time = time.time()
    latest_time = start_time
    preproc_class = get_preprocessor_class(preprocessor_name)
    preprocessor = preproc_class(target, categ_features)

    # loading
    logging.info('Loading raw dataset: {}...'.format(dataset_name))
    frame = load_raw_dataset(dataset_name, target, preproc_class, working_folder, s3_bucket, s3_folder, headers, verbose, clean)
    preprocessor.initialize(frame)
    latest_time = timing('loading data', latest_time)

    # pre-processing
    logging.info('Pre-processing...')
    frame = preprocessor.pre_process(frame)
    latest_time = timing('pre-processing', latest_time)

    # splitting
    logging.info('Splitting dataset...')
    frames_dict = preproc_class.split_datasets(frame, [("train", TRAIN_PROPORTION),
                                                       ("valid", VALID_PROPORTION),
                                                       ("test", TEST_PROPORTION)], target)
    latest_time = timing('splitting', latest_time)

    # normalizing
    if scale:
        logging.info('Normalizing datasets...')
        frames_dict = preprocessor.normalize(frames_dict)
        latest_time = timing('normalizing', latest_time)

    # one-hot-encoding
    if categ_encoding:
        logging.info('Encoding categorical features...')
        frames_dict = preprocessor.categorical_encoding(frames_dict)
        latest_time = timing('encoding', latest_time)

    preprocessor.log_dataset_features(frames_dict)

    # saving
    logging.info('Saving datasets...')
    save_datasets(frames_dict, working_folder, dataset_name + '_preprocessed', s3_bucket, s3_folder, verbose)
    latest_time = timing('saving', latest_time)

    timing('total time', start_time)


# =========================================================
# MAIN
# =========================================================
if __name__ == '__main__':
    # Input parsing
    parser = argparse.ArgumentParser(description='PRE-PROCESSING SCRIPT ')
    parser.add_argument('-d', dest='dataset_name', type=str, help='input dataset name (without extension)', required=True)
    parser.add_argument('-f', dest='data_folder', type=str, help='data folder (default data)', default='data')
    parser.add_argument('--infra-s3', dest='infra_s3', type=str, help='configuration of the S3 bucket from infra_s3.yaml', default=None)
    parser.add_argument('-t', dest='target', type=str, help='feature name for the labels, or index if no headers', required=True)
    parser.add_argument('-p', dest='preprocessor', type=str, help='preprocessor name', default='')
    parser.add_argument('-v', dest='verbose', help='verbose mode', action='store_true')
    parser.add_argument('--headers', dest='headers', help='parse headers', action='store_true')
    parser.add_argument('--scale', dest='scale', help='subtract the mean and divide by the std', action='store_true')
    parser.add_argument('--encode', dest='categ_encoding', help='one-hot encoding of categorical variables (label encoding for H2O)', action='store_true')
    parser.add_argument('--categ', nargs='+', dest='categ_features', help='list of categorical features, encoded if --encoding is specified', default=None)
    parser.add_argument('--clean', dest='clean', help='ignore previously generated local files and erase intermediate files to regenerate them', action='store_true')

    args = parser.parse_args()

    infra_s3 = load_infra_s3(args.infra_s3)

    s3_bucket, s3_folder = None, None
    if infra_s3 is not None:
        s3_bucket = infra_s3['s3_bucket']
        s3_folder = infra_s3['s3_folder_data']

    prepare_datasets(args.dataset_name, args.data_folder, s3_bucket, s3_folder, args.target,
                     args.headers, args.verbose, args.clean, args.preprocessor.lower(),
                     args.scale, args.categ_encoding, args.categ_features)
