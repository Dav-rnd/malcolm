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
from preprocessors.preprocessing import Preprocessor


class PreprocessorAbalone(Preprocessor):
    DATE_FEATURES = []

    def __init__(self, target: str, categ_features: list=None):
        super().__init__(target, categ_features)

    def pre_process(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        General preprocessing, applied without considering whether the input data
        is for training, testing or validation purposes. Split has not occurred yet
        """
        # frame = rows_filtering(frame)
        # frame = feature_dropping(frame)
        # frame = feature_values_fixing(frame)

        # frame = extreme_values_handling(frame, [])
        # missing_value_imputation(frame, [])

        # data_type_conversion(frame)
        # frame = feature_engineering(frame, self.GENERATE_USER_FEATURES)
        # feature_renaming(frame)

        return frame
