# *********************************************************
# PREPROCESSING - SAVE_DATASETS
# (C) Alexandre Salch 2019
# (C) David Renaudie 2019
# *********************************************************

# =========================================================
# IMPORTS
# =========================================================
import pandas as pd
from typing import Dict
import logging
import os
import boto3

# =========================================================
# GLOBALS
# =========================================================
pd.set_option('io.hdf.default.format', 'fixed')


# =========================================================
# FUNCTIONS
# =========================================================
def save_datasets(frames_dict: Dict[str, pd.DataFrame], working_folder: str, output_name: str,
                  s3_bucket: str=None, s3_folder: str=None, verbose: bool=False) -> None:
    output_filename = output_name + '.h5'
    if verbose:
        df = next(iter(frames_dict.values()))
        logging.info('Saved dataset ({} observations, {} features):\n{}\nColumn types:\n{}'.format(
                     df.shape[0], df.shape[1], df.head(5), df.dtypes))
    local_s5_file = os.path.join(working_folder, output_filename)
    save_datasets_to_h5(frames_dict, local_s5_file)

    if s3_bucket is not None:
        s3_key = "/".join([s3_folder, output_filename])
        upload_file_to_s3(local_s5_file, s3_bucket, s3_key)


def save_datasets_to_h5(frames_dict: Dict[str, pd.DataFrame], file: str) -> None:
    logging.info('saving h5 to: {}'.format(file))

    with pd.HDFStore(file, 'w', complevel=4) as h5_store:
        for i, (dataframe_name, dataframe) in enumerate(frames_dict.items()):
            logging.info('{}/{}: saving dataframe: {}...'.format(i + 1, len(frames_dict), dataframe_name))
            h5_store[dataframe_name] = dataframe
    logging.info('saving h5 done.')


def upload_file_to_s3(file: str, s3_bucket: str, s3_key: str) -> None:
    logging.info('uploading {} to s3: {}:{}...'.format(file, s3_bucket, s3_key))
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(file, s3_bucket, s3_key)
    logging.info('uploading done.')
