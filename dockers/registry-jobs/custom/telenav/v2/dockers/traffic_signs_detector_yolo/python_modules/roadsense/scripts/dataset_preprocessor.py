import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
from glob import glob
from keras.utils.np_utils import to_categorical

import roadsense.scripts.utils as utils
from roadsense.scripts.config import ConfigParams as cp, HazardType as HazardType, Column

import apollo_python_common.io_utils as io_utils

tqdm.pandas()


class DatasetPreprocessor:
    COLS_OF_INTEREST = [Column.ALL_FEATURES, Column.HAZARD,
                        Column.TRIP_ID, Column.RAW_HAZARD,
                        Column.LAT, Column.LON,Column.IMAGE_INDEX,
                        Column.HAZARD_LAT, Column.HAZARD_LON]

    def __init__(self, train_config):
        self.train_config = train_config
        self.dataset_config = io_utils.json_load(train_config[cp.DATASET_CONFIG_PATH])

    def __get_features_and_label(self, df):
        features = np.asarray(df[Column.ALL_FEATURES].tolist()).astype(np.float32)
        labels = np.asarray(df[Column.HAZARD].tolist())

        return features, labels

    def process_folder(self, drive_folder, class_balance_factor=-1):

        data_df = utils.read_sensor_df(drive_folder)[self.COLS_OF_INTEREST].sort_index()
        data_df.loc[:, Column.ALL_FEATURES] = data_df.loc[:, Column.ALL_FEATURES].apply(lambda arr: arr.flatten())

        if class_balance_factor != -1:
            hazard_count_df = data_df[Column.HAZARD].value_counts()
            hazard_count_df = hazard_count_df[hazard_count_df.index != HazardType.CLEAR]

            biggest_hazard_count = hazard_count_df[0] if len(hazard_count_df) > 0 else 1
            nr_clear = max(1, int(class_balance_factor * biggest_hazard_count))
            data_df = pd.concat([data_df[data_df[Column.HAZARD] == HazardType.CLEAR][:nr_clear],
                                 data_df[data_df[Column.HAZARD] != HazardType.CLEAR]]
                                )

        X, y = self.__get_features_and_label(data_df)

        return X, y, data_df

    def __process_folders(self, drive_folders, class_balance_factor=-1):

        X_list, y_list, df_list = [], [], []

        for drive_folder in tqdm(drive_folders):
            print(os.path.basename(drive_folder))
            X, y, data_df = self.process_folder(drive_folder, class_balance_factor)

            X_list.append(X)
            y_list.append(y)
            df_list.append(data_df)

        return shuffle(np.vstack(X_list), np.hstack(y_list), pd.concat(df_list), random_state=0)

    def to_ohe(self, label_data, class_2_index):
        label_data_indexed = [class_2_index[label] for label in label_data]
        label_data_ohe = to_categorical(np.asarray(label_data_indexed))
        return label_data_ohe

    def __get_drive_folders(self):
        return sorted([drive_folder for drive_folder in
                       glob(os.path.join(self.dataset_config[cp.DATASET_BASE_PATH],
                                         utils.get_dataset_folder_name(self.dataset_config)) + "/*")
                       if "drive" in drive_folder])

    def __get_train_test_drive_folders(self):
        drive_folders = self.__get_drive_folders()
        test_drive_day = self.train_config[cp.TEST_DRIVE_DAY]

        test_drive_folders = [drive_folder for drive_folder in drive_folders if test_drive_day in drive_folder]
        train_drive_folders = sorted(list(set(drive_folders) - set(test_drive_folders)))

        return train_drive_folders, test_drive_folders

    def keep_hazards_of_interest(self, y_arr):
        hazards_of_interest = self.train_config[cp.KEPT_HAZARDS]
        return np.asarray([y if y in hazards_of_interest else HazardType.CLEAR for y in y_arr])

    def __keep_hazards_of_interest_for_dataset(self, y_train, y_test):

        y_train = self.keep_hazards_of_interest(y_train)
        y_test = self.keep_hazards_of_interest(y_test)

        return y_train, y_test

    def __transform_to_ohe(self, y_train, y_test):
        y_train_ohe = self.to_ohe(y_train, self.train_config[cp.CLASS_2_INDEX])
        y_test_ohe = self.to_ohe(y_test, self.train_config[cp.CLASS_2_INDEX])

        return y_train_ohe, y_test_ohe

    def get_train_test_data(self):

        print("Loading data...")

        train_drive_folders, test_drive_folders = self.__get_train_test_drive_folders()

        X_train, y_train, _, = self.__process_folders(train_drive_folders,
                                                      self.train_config[cp.TRAIN_CLASS_BALANCE_FACTOR])

        X_test, y_test, test_df = self.__process_folders(test_drive_folders)

        y_train, y_test = self.__keep_hazards_of_interest_for_dataset(y_train, y_test)

        y_train_ohe, y_test_ohe = self.__transform_to_ohe(y_train, y_test)

        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)

        print("\n=======Train=======")
        print(pd.Series(y_train).value_counts(normalize=False))
        print(pd.Series(y_train).value_counts(normalize=True))

        print("\n=======Test=======")
        print(pd.Series(y_test).value_counts(normalize=False))
        print(pd.Series(y_test).value_counts(normalize=True))

        return X_train, y_train_ohe, X_test, y_test_ohe, test_df

    def filter_single_trip(self, test_df, trip_prefix):
        return test_df[test_df[Column.TRIP_ID].str.contains(trip_prefix)].sort_index()

    def add_preds_to_df(self, test_df, y_pred_proba, best_conf_thresh):
        y_pred = utils.keep_high_conf_hazards(y_pred_proba, best_conf_thresh, self.train_config[cp.CLASS_2_INDEX])
        test_df[Column.PRED] = pd.Series(y_pred, index=test_df.index)
        return test_df
