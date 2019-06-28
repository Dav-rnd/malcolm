# *********************************************************
# PREPROCESSING - LOAD_DATASET
# (C) Alexandre Salch 2019
# (C) David Renaudie 2019
# (C) Remi Domingues 2019
# *********************************************************

# =========================================================
# IMPORTS
# =========================================================
import pandas as pd
import logging
import os
import bz2
import boto3
import numpy as np
from typing import Tuple

from preprocessors.preprocessing import Preprocessor
from save_datasets import save_datasets_to_h5
# =========================================================
# GLOBALS
# =========================================================


# =========================================================
# FUNCTIONS
# =========================================================

def load(load_order: list, reverse: bool=False) -> pd.DataFrame:
    """ Execute loading functions in the given order
    If reverse is true, e.g. in a cleaning setting when we wish to start
    from base files, the loading order is reversed
    """
    if reverse:
        load_order = list(reversed(load_order))

    data = None
    for loading_fct in load_order:
        data = loading_fct()
        if data is not None:
            break

    return data


def load_raw_dataset(dataset_name: str, target_name: str, preprocessor, working_folder: str, s3_bucket: str=None, s3_folder: str=None,
                     headers: bool=False, verbose: bool=False, clean: bool=False) -> pd.DataFrame:
    local_filename_h5 = os.path.join(working_folder, dataset_name + '.h5')
    local_filename_csv = os.path.join(working_folder, dataset_name + '.csv')
    local_filename_bz2 = os.path.join(working_folder, dataset_name + '.csv.bz2')

    def _load_local_h5():
        if os.path.isfile(local_filename_h5):
            return load_h5_single(local_filename_h5)
        return None

    def _load_local_csv():
        if os.path.isfile(local_filename_csv):
            frame = load_csv(local_filename_csv, target_name, preprocessor, headers)
            save_datasets_to_h5({dataset_name: frame}, local_filename_h5)
            return frame
        return None

    def _load_local_bz2():
        if os.path.isfile(local_filename_bz2):
            decompress_bz2(local_filename_bz2, local_filename_csv)
            frame = load_csv(local_filename_csv, target_name, preprocessor, headers)
            save_datasets_to_h5({dataset_name: frame}, local_filename_h5)
            return frame
        return None

    def _load_s3_csv_bz2():
        if s3_bucket is not None and s3_folder is not None:
            cloud_filename_s3 = s3_folder + "/" + dataset_name
            if not download_file_from_s3(local_filename_csv, s3_bucket, cloud_filename_s3 + '.csv'):
                if download_file_from_s3(local_filename_bz2, s3_bucket, cloud_filename_s3 + '.csv.bz2'):
                    decompress_bz2(local_filename_bz2, local_filename_csv)
                else:
                    return None

            frame = load_csv(local_filename_csv, target_name, preprocessor, headers)
            save_datasets_to_h5({dataset_name: frame}, local_filename_h5)
            return frame
        return None

    frame = load([_load_local_h5, _load_local_csv, _load_local_bz2, _load_s3_csv_bz2], reverse=clean)

    if frame is None:
        raise FileNotFoundError('Cannot locate dataset: {}'.format(dataset_name))

    # $ is a special character that we keep only to separate column name and the one hot encoded value (cname_val)
    frame.rename(columns=lambda c: c.replace(Preprocessor.OHE_NAME_SEP, '_'), inplace=True)

    if verbose:
        logging.info('Data schema ({} observations, {} features):\n{}'.format(len(frame), len(frame.columns), frame.dtypes))
        logging.info('Data statistics:\n{}'.format(frame.describe()))

    if frame.isnull().values.any():
        logging.warning('***** The loaded dataframe contains NaN values. This could be caused by rows with missing fields. *****')

    return frame


def filter_categ(frames: list, onehot: bool, target: str):
    """
    onehot: if true, categorical variables will be one-hot encoded (sklearn, SageMaker)
            Otherwise, they will be encoded as unique integer IDs (H2O)
    """
    def col_types(cols: list, enum_cols: list):
        """
        Return a list of types: ['numeric'|'enum'] * n_cols
        cols is the complete list of columns enum_cols are a subset of enum columns
        """
        return {c: ('enum' if c in enum_cols else 'numeric') for c in cols}

    def discarded_cols(df: pd.DataFrame, keep_one_hot: bool):
        from preprocessors.preprocessing import Preprocessor
        # One-hot encoded categorical colums
        one_hot_cols = [c for c in df.columns if Preprocessor.OHE_NAME_SEP in c]
        # Categorical colums encoded as integer IDs
        # Retrieve unique original column names for one hot encoded names (masking)
        categ_cols = list(set(map(lambda c: c.split(Preprocessor.OHE_NAME_SEP)[0], one_hot_cols)))

        if keep_one_hot:
            # Categ columns will be discarded, types are only numerics except for target
            return categ_cols, col_types([c for c in df.columns if c not in categ_cols], [target])

        # One-hot encoded columns will be discarded, categ_cols and target are enums
        return one_hot_cols, col_types([c for c in df.columns if c not in one_hot_cols], categ_cols + [target])

    cols, types = discarded_cols(frames[0], onehot)
    logging.info('Discarding features (keep_one_hot={}): {}'.format(onehot,
        set([f if Preprocessor.OHE_NAME_SEP not in f else f.split(Preprocessor.OHE_NAME_SEP)[0] + Preprocessor.OHE_NAME_SEP + '*' for f in cols])))

    for i, df in enumerate(frames):
        frames[i] = df.drop(cols, axis=1)

    return frames, types


def load_preproc_dataset(dataset_name: str, working_folder: str, s3_bucket: str=None, s3_folder: str=None,
        clean: bool=False, onehot=True, target: str=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    onehot: if true, categorical variables will be one-hot encoded (sklearn, SageMaker)
            Otherwise, they will be encoded as unique integer IDs (H2O)
    """
    local_filename_h5 = os.path.join(working_folder, dataset_name + '.h5')

    def _load_local_h5():
        if os.path.isfile(local_filename_h5):
            logging.info('Found local dataset {}'.format(local_filename_h5))
            training, validation, testing = load_h5_multiple(local_filename_h5)
            (training, validation, testing), col_types = filter_categ([training, validation, testing], onehot, target)
            return training, validation, testing, col_types
        return None

    def _load_s3_h5():
        if s3_bucket is not None and s3_folder is not None:
            cloud_filename_s3 = s3_folder + "/" + dataset_name + '.h5'
            download_file_from_s3(local_filename_h5, s3_bucket, cloud_filename_s3)
            training, validation, testing = load_h5_multiple(local_filename_h5)
            (training, validation, testing), col_types = filter_categ([training, validation, testing], onehot, target)
            return training, validation, testing, col_types
        return None

    res = load([_load_local_h5, _load_s3_h5], reverse=clean)

    if res is None:
        logging.error('file not found: {}'.format(dataset_name))
        raise FileNotFoundError

    # training, validation, testing, col_types
    return res


def load_h5_single(file: str) -> pd.DataFrame:
    logging.info('Loading single dataframe from h5...'.format(file))
    frame = pd.read_hdf(file)
    return pd.DataFrame(frame)


def load_h5_multiple(file: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logging.info('Loading multiple dataframes from h5...'.format(file))
    with pd.HDFStore(file, mode='r') as store:
        training = pd.DataFrame(store['train'])
        validation = pd.DataFrame(store['valid'])
        testing = pd.DataFrame(store['test'])
    return training, validation, testing


def load_csv(file: str, target: str, preprocessor, headers: bool=False) -> pd.DataFrame:
    logging.info('Loading from csv: {}...'.format(file))
    try:
        df = pd.read_csv(file, sep=',', low_memory=False, na_values='na', header='infer' if headers else None,
                         dtype={target: str}, parse_dates=preprocessor.DATE_FEATURES)
    except Exception:
        # Note: low_memory=True will cause the file to be processed in chunk, resulting in possible mixed type inference
        # Yet, the separator cannot be automatically inferred (sep=None) if low_memory is False
        df = pd.read_csv(file, sep=None, low_memory=True, na_values='na', header='infer' if headers else None,
                         dtype={target: str}, parse_dates=preprocessor.DATE_FEATURES)
    df.columns = df.columns.astype(str)
    return df


def decompress_bz2(input_file: str, output_file: str) -> None:
    logging.info('Decompressing: {}...'.format(input_file))
    with open(input_file, 'rb') as source, open(output_file, 'wb') as destination:
        destination.write(bz2.decompress(source.read()))


def download_file_from_s3(file: str, s3_bucket: str, s3_key: str) -> bool:
    logging.info('downloading from s3: {}:{}...'.format(s3_bucket, s3_key))
    if is_s3_file(s3_bucket, s3_key):
        os.makedirs(os.path.dirname(file), exist_ok=True)
        s3 = boto3.resource('s3')
        s3.meta.client.download_file(s3_bucket, s3_key, file)
        logging.info('downloading done.')
        return True
    else:
        logging.info('file {}/{} not found on S3'.format(s3_bucket, s3_key))
        return False


def is_s3_file(s3_bucket: str, s3_filepath: str, verbose: bool=True) -> bool:
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(s3_bucket)
    try:
        objects = list(bucket.objects.filter(Prefix=s3_filepath))
    except Exception:
        raise ValueError('Could not connect to AWS. Please check your credentials and region.')

    if len(objects) > 0:
        if verbose:
            logging.info('Existing file {}/{}'.format(s3_bucket, s3_filepath))
        return True

    if verbose:
        logging.info('File not found: {}/{}'.format(s3_bucket, s3_filepath))
    return False
