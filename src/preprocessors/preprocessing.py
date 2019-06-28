# *********************************************************
# PREPROCESSING - PREPROCESSING
# (C) Alexandre Salch 2019
# (C) David Renaudie 2019
# (C) Remi Domingues 2019
# *********************************************************

# =========================================================
# IMPORTS
# =========================================================
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple
from collections import Counter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def imputation_missing_values_by_mode(frame: pd.DataFrame, feature: str):
    nb_missing_values = frame[feature].isnull().sum()
    most_freq = frame[feature].dropna().mode()[0]
    frame[feature].fillna(most_freq, inplace=True)
    logging.info('Most frequent value for {}: {}. Replaced {} missing values.'.format(feature, most_freq, nb_missing_values))


def missing_value_imputation(frame: pd.DataFrame, features: list):
    for feature in features:
        imputation_missing_values_by_mode(frame, feature)


def filter_out_extreme_values_iqr(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    logging.info('Filtering extreme values for {}'.format(feature))
    q1 = df[feature].quantile(.25)
    q3 = df[feature].quantile(.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    higher = q3 + 1.5 * iqr
    logging.info('q1: {:.2f} / q3: {:.2f} / lower: {:.2f} / higher: {:.2f}'.format(q1, q3, lower, higher))
    mask = df[feature].between(lower, higher, inclusive=True)

    return df[mask]


def extreme_values_handling(frame: pd.DataFrame, features_to_filter_extreme_values: list) -> pd.DataFrame:
    for feature in features_to_filter_extreme_values:
        frame = filter_out_extreme_values_iqr(frame, feature)

    return frame


class Preprocessor(object):
    DATE_FEATURES = []
    UNKNOWN_CLASS_LABEL = chr(32768)
    # Separator for one-hot encoded column names (cname<sep>value)
    OHE_NAME_SEP = '$'

    def __init__(self, target: str, categ_cols: list=None):
        self.fitted_num = False
        self.num_cols = None
        self.fitted_categ = False
        self.target = target
        self.n_classes = None
        self.categ_cols = categ_cols

    def initialize(self, df: pd.DataFrame):
        if self.n_classes is None:
            self.n_classes = df[self.target].nunique()

        if self.categ_cols is None:
            self.categ_cols = list(df.dtypes[df.dtypes == object].index)
        else:
            missing_features = [f for f in self.categ_cols if f not in df.columns]
            if missing_features:
                raise ValueError('The specified categorical features do not exist: {}\nExisting features are: {}'.format(
                                 missing_features, list(df.columns)))

        if self.target in self.categ_cols:
            self.categ_cols.remove(self.target)

        if self.num_cols is None:
            self.num_cols = df.drop(self.target, axis=1)._get_numeric_data().columns
            self.categ_cols_to_convert = [f for f in self.num_cols if f in self.categ_cols]
            if self.target not in self.categ_cols_to_convert:
                self.categ_cols_to_convert.append(self.target)
            self.num_cols = [f for f in self.num_cols if f not in self.categ_cols]

    def pre_process(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        General preprocessing, applied without considering whether the input data
        is for training, testing or validation purposes. Split has not occurred yet
        """
        logging.info('Preprocessing step not implemented')

        return frame

    def _categ_num_conversion(self, df):
        """ Numerical columns which should be interpreted as categoricals are converted in strings """
        df.loc[:, self.categ_cols_to_convert] = df[self.categ_cols_to_convert].astype(str)  # 'category' also available

    def _train_label_encoder(self, df):
        """ Train the label encoders on each feature to learn a mapping str => int ID """
        self.encoders = [LabelEncoder().fit(df[col]) for col in self.categ_cols + [self.target]]

        # Extend the labels with an 'unknown' class for all new categorical values (unused if --new-categ ignore)
        for encoder in self.encoders:
            encoder.classes_ = np.append(encoder.classes_, [Preprocessor.UNKNOWN_CLASS_LABEL])
            if sorted(encoder.classes_)[-1] != Preprocessor.UNKNOWN_CLASS_LABEL:
                raise ValueError('Unknown class label. \'{}\' should have an encoding value higher than \'{}\'. Please update the Preprocessor.UNKNOWN_CLASS_LABEL value accordingly.'.format(
                                 Preprocessor.UNKNOWN_CLASS_LABEL, sorted(encoder.classes_)[-1]))

    def _label_encoding(self, df, encode_unknown=True):
        """
        Encode string labels into integer IDs
        Return a df composed only of label encoded features (i.e. copy of the DF with int IDs only)
        """
        data = {}
        for col, encoder in zip(self.categ_cols + [self.target], self.encoders):
            # All new categ values are encoded, if df is not the training set
            if encode_unknown:
                # Replace new categorical values by <unknown> class. Encoding would fail otherwise
                df.loc[~df[col].isin(encoder.classes_), col] = Preprocessor.UNKNOWN_CLASS_LABEL

            # Encode categorical features into integer IDs
            data[col] = encoder.transform(df[col]).astype(int)

        return pd.DataFrame(data, index=df.index)

    def _train_one_hot_encoder(self, df):
        """ Train the one hot encoders on each categorical feature """
        self.one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.one_hot_encoder.fit(df[self.categ_cols])

    def _one_hot_encoding(self, df):
        """
        One-hot encoding of categorical data: each categorical feature is replaced by several boolean features
        Return a df composed only of one-hot encoded features (i.e. copy of the DF with bool features only)
        The target column is not one-hot encoded, and thus not returned in the final dataframe
        """
        def exclude_col(class_name):
            """ No column dedicated to new categories when we ignore new values """
            return class_name == Preprocessor.UNKNOWN_CLASS_LABEL

        categ = self.one_hot_encoder.transform(df[self.categ_cols]).astype(int)

        # No one-hot encoding for target
        col_names = np.concatenate([[str(colname) + self.OHE_NAME_SEP + (value if value != Preprocessor.UNKNOWN_CLASS_LABEL else 'n/a')
                                    for value in enc.classes_ if not exclude_col(value)]
                                    for colname, enc in zip(self.categ_cols, self.encoders[:-1])])
        return pd.DataFrame(categ, columns=col_names, index=df.index)

    def _merge_encoded_features(self, df_original, df_label_enc, df_onehot):
        """
        Return a dataframe composed of:
        - the one-hot encoded features (<feature>$<value>)
        - the original (str) categorical features
        - the numerical features
        - the target features encoded as inter IDs
        """
        return pd.concat([df_onehot, df_original.drop(self.target, axis=1), df_label_enc[self.target]], axis=1)

    def categorical_encoding(self, frames_dict: Dict[str, pd.DataFrame]):
        """
        Encodes categorical features into numerical IDs, then IDs into boolean one-hot encoded features
        Labels are only encoded into IDs
        """
        frames_label_enc = {}
        frames_onehot = {}

        logging.info('Categorical features to encode: {}'.format(self.categ_cols))

        if len(self.categ_cols) > 0:
            for df in frames_dict.values():
                self._categ_num_conversion(df)

            if not self.fitted_categ:
                self._train_label_encoder(frames_dict['train'])

            for df_name, df in frames_dict.items():
                frames_label_enc[df_name] = self._label_encoding(df, encode_unknown=df_name!='train')

            if not self.fitted_categ:
                self._train_one_hot_encoder(frames_label_enc['train'])
                self.fitted_categ = True

            # One-hot encoding: int ID => several bool features
            for df_name, df in frames_label_enc.items():
                frames_onehot[df_name] = self._one_hot_encoding(df)

            for df_name in frames_dict.keys():
                frames_dict[df_name] = self._merge_encoded_features(frames_dict[df_name], frames_label_enc[df_name], frames_onehot[df_name])

            logging.info('{} categorical features ({}) were encoded into {} boolean features'.format(
                         len(self.categ_cols), self.categ_cols, len(frames_onehot['train'].columns)))

        return frames_dict

    def normalize(self, frames_dict: Dict[str, pd.DataFrame]):
        """ Scale the input data by subtracting the mean and dividing by the standard deviation """

        if len(self.num_cols) > 0:
            if not self.fitted_num:
                df_train = frames_dict["train"]
                self.mean, self.std = df_train[self.num_cols].mean(), df_train[self.num_cols].std()
                self.std[self.std == 0] = 1
                self.fitted_num = True

            for df_name in frames_dict.keys():
                frames_dict[df_name][self.num_cols] = (frames_dict[df_name][self.num_cols] - self.mean) / self.std
                # df.loc[:, self.num_cols] = (df[self.num_cols] - self.mean) / self.std

        return frames_dict

    def log_dataset_features(self, frames_dict):
        logging.info('Preprocessed dataset: {} features ({} numerical, {} categorical into {} booleans), 1 target ({} classes)'.format(
            len(frames_dict['train'].columns) - 1,
            len(self.num_cols), 0 if not self.fitted_categ else len(self.categ_cols),
            0 if not self.fitted_categ else len(frames_dict['train'].columns)- 1 - len(self.num_cols),
            self.n_classes))

    @staticmethod
    def split_datasets(frame: pd.DataFrame, proportions: List[Tuple[str, float]], target: str) -> Dict[str, pd.DataFrame]:
        if target is not None:
            counts = Counter(frame[target]).most_common()
            logging.info('Total class distribution (ID, #): {}'.format(counts))
            logging.info('Total class proportions (ID, %): {}'.format([(cid, np.round(cnt / len(frame) * 100, 2)) for (cid, cnt) in counts]))
        else:
            logging.info('Class distribution not available (no target)')

        frame = frame.sample(frac=1)  # Shuffle

        start = 0

        frames_dict = {}

        for i in range(len(proportions)):
            ds_size = int(proportions[i][1] * len(frame))
            if i != len(proportions) - 1:
                end = start + ds_size
                current_frame = frame[start:end]
                start = end
            else:
                current_frame = frame[start:]

            if target is not None:
                counts = Counter(current_frame[target]).most_common()
                logging.info('{} class proportions (ID, %): {}'.format(proportions[i][0],
                    [(cid, np.round(cnt / len(current_frame) * 100, 2)) for (cid, cnt) in counts]))

            frames_dict[proportions[i][0]] = current_frame

        return frames_dict


if __name__ == '__main__':
    data_train = {'Name': ['Tom', 'nick', 'krish', 'jack'], 'Age': [20, 21, 19, 18], 'b': [0, 1, 0, 0], 'c': ['a', 'b', 'b', 'a']}
    data_test = {'Name': ['Tom', 'nick', 'krish', 'random'], 'Age': [20, 21, 19, 18], 'b': [0, 1, 0, 3], 'c': ['a', 'b', 'b', 'a']}
    t = Preprocessor('c', categ_cols=['Name', 'b'])
    frames_dict = {'train': pd.DataFrame(data_train), 'test': pd.DataFrame(data_test)}
    print(frames_dict['test'])

    print('Normalize...')
    t.initialize(frames_dict['train'])
    t.normalize(frames_dict)
    print(frames_dict['test'])

    print('Encoding...')
    t.categorical_encoding(frames_dict)
    print(frames_dict['train'])
    print(frames_dict['test'])
