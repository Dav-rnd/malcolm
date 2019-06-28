# *********************************************************
# COMMON UTILS
# (C) David Renaudie 2019
# *********************************************************
import yaml
import time
import logging


def timing(step_name, previous_time):
    new_time = time.time()
    logging.info('TIMING: {}: {:.0f} ms.'.format(step_name, 1000 * (new_time - previous_time)))
    return new_time


def load_config(config_filename: str):
    # load configuration file
    config_file = 'config/{}'.format(config_filename)
    with open(config_file, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


def load_infra_s3(s_infra_s3):
    """
    Return a dictionary corresponding to the S3 configuration located at
    config/infra_s3.yaml:<s_infra_s3>
    """
    if s_infra_s3 is None:
        return None

    s3_config_filename = 'infra_s3.yaml'
    config_s3 = load_config(s3_config_filename)
    if s_infra_s3 not in config_s3:
        raise ValueError('Cannot find S3 config \'{}\' in config file {}'.format(s_infra_s3, s3_config_filename))
    infra_s3 = config_s3[s_infra_s3]

    if infra_s3['s3_folder_data'][-1] == "/":
        infra_s3['s3_folder_data'][-1] = infra_s3['s3_folder_data'][-1][:-1]

    return infra_s3


def load_infra_sm(s_infra_sm):
    """
    Return a dictionary corresponding to the SageMaker configuration located at
    config/infra_sm.yaml:<s_infra_s3>
    """
    if s_infra_sm is None:
        return None

    sm_config_filename = 'infra_sm.yaml'
    config_sm = load_config(sm_config_filename)
    if s_infra_sm not in config_sm:
        raise ValueError('Cannot find SageMaker config \'{}\' in config file {}'.format(s_infra_sm, sm_config_filename))
    return config_sm[s_infra_sm]
