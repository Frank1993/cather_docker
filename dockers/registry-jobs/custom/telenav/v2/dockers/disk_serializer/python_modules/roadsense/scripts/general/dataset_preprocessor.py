import multiprocessing
import os
from functools import partial
from glob import glob
from multiprocessing import Pool

import apollo_python_common.io_utils as io_utils
import numpy as np
import pandas as pd
import roadsense.scripts.general.utils as utils
from keras.utils.np_utils import to_categorical
from roadsense.scripts.general.config import ConfigParams as cp, HazardType as HazardType, Column
from sklearn.utils import shuffle


class DatasetPreprocessor:
    COLS_OF_INTEREST = [Column.ALL_FEATURES, Column.HAZARD,
                        Column.TRIP_ID, Column.RAW_HAZARD,
                        Column.LAT, Column.LON,Column.IMAGE_INDEX,
                        Column.HAZARD_LAT, Column.HAZARD_LON,
                        Column.WAY_ID,Column.FROM_NODE_ID,
                        Column.TO_NODE_ID,Column.MATCHED_LAT,Column.MATCHED_LON, Column.COMPUTED_HEADING,
                        Column.SECTION_FROM_NODE_ID,Column.SECTION_FROM_NODE_ID_LAT,Column.SECTION_FROM_NODE_ID_LON,
                        Column.SECTION_TO_NODE_ID,Column.SECTION_TO_NODE_ID_LAT,Column.SECTION_TO_NODE_ID_LON
                       ]

    def __init__(self, train_config):
        self.train_config = dict(train_config)
        self.dataset_config = dict(io_utils.json_load(train_config[cp.DATASET_CONFIG_PATH]))

    def _get_features_and_label(self, df):
        features = np.asarray(df[Column.ALL_FEATURES].tolist()).astype(np.float32)
        labels = np.asarray(df[Column.HAZARD].tolist())

        return features, labels

    def process_folder(self, drive_folder, class_balance_factor=-1):
        data_df = utils.read_sensor_df(drive_folder)
        kept_cols = list(set(self.COLS_OF_INTEREST).intersection(set(data_df.columns)))    
        data_df = data_df[kept_cols].sort_index()
        data_df[Column.ALL_FEATURES] = data_df[Column.ALL_FEATURES].apply(lambda arr: arr.flatten())

        if class_balance_factor != -1:
            hazard_count_df = data_df[Column.HAZARD].value_counts()
            hazard_count_df = hazard_count_df[hazard_count_df.index != HazardType.CLEAR]

            biggest_hazard_count = hazard_count_df[0] if len(hazard_count_df) > 0 else 1
            nr_clear = max(1, int(class_balance_factor * biggest_hazard_count))
            data_df = pd.concat([data_df[data_df[Column.HAZARD] == HazardType.CLEAR][:nr_clear],
                                 data_df[data_df[Column.HAZARD] != HazardType.CLEAR]]
                                )
        X, y = self._get_features_and_label(data_df)
        return X, y, data_df.drop([Column.ALL_FEATURES],axis=1)

    def process_folders(self, drive_folders, class_balance_factor=-1):
        
        pool = Pool(multiprocessing.cpu_count() // 2)
        
        results = pool.map(partial(self.process_folder, class_balance_factor = class_balance_factor), 
                           drive_folders)
        
        X_list = [x for x,_,_ in results]
        y_list = [y for _,y,_ in results]
        df_list = [data_df for _,_,data_df in results]
        
        return shuffle(np.vstack(X_list), np.hstack(y_list), pd.concat(df_list), random_state=0)

    def to_ohe(self, label_data, class_2_index):
        label_data_indexed = [class_2_index[label] for label in label_data]
        label_data_ohe = to_categorical(np.asarray(label_data_indexed))
        return label_data_ohe

    def _get_drive_folders(self):
        return sorted([drive_folder for drive_folder in
                       glob(os.path.join(self.dataset_config[cp.DATASET_BASE_PATH],
                                         utils.get_dataset_folder_name(self.dataset_config)) + "/*")
                       if "drive" in drive_folder])

    def _get_train_test_drive_folders(self):
        drive_folders = self._get_drive_folders()
        test_drive_days = self.train_config[cp.TEST_DRIVE_DAYS]

        test_drive_folders = []
        for test_drive_day in test_drive_days:
            for drive_folder in drive_folders:
                if test_drive_day in drive_folder:
                    test_drive_folders.append(drive_folder)
                    break
            
        train_drive_folders = sorted(list(set(drive_folders) - set(test_drive_folders)))
     
        return train_drive_folders, test_drive_folders


        
    def keep_hazards_of_interest(self, y_arr):
        hazards_of_interest = self.train_config[cp.KEPT_HAZARDS]
        return np.asarray([y if y in hazards_of_interest else HazardType.CLEAR for y in y_arr])

    def _keep_hazards_of_interest_for_dataset(self, y_train, y_test):

        y_train = self.keep_hazards_of_interest(y_train)
        y_test = self.keep_hazards_of_interest(y_test)

        return y_train, y_test

    def _transform_to_ohe(self, y_train, y_test):
        y_train_ohe = self.to_ohe(y_train, self.train_config[cp.CLASS_2_INDEX])
        y_test_ohe = self.to_ohe(y_test, self.train_config[cp.CLASS_2_INDEX])

        return y_train_ohe, y_test_ohe

    def _print_data_stats(self, X_train,y_train, X_test, y_test):
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)

        print("\n=======Train=======")
        print(pd.Series(y_train).value_counts(normalize=False))
        print(pd.Series(y_train).value_counts(normalize=True))

        print("\n=======Test=======")
        print(pd.Series(y_test).value_counts(normalize=False))
        print(pd.Series(y_test).value_counts(normalize=True))

        
    def get_train_test_data(self):

        print("Loading data...")

        train_drive_folders, test_drive_folders = self._get_train_test_drive_folders()
        print(f"Train folders {len(train_drive_folders)}")
        print(f"Test folders {len(test_drive_folders)}")
        
        X_train, y_train, _, = self.process_folders(train_drive_folders,
                                                    self.train_config[cp.TRAIN_CLASS_BALANCE_FACTOR])

        X_test, y_test, test_df = self.process_folders(test_drive_folders)
        y_train, y_test = self._keep_hazards_of_interest_for_dataset(y_train, y_test)
        y_train_ohe, y_test_ohe = self._transform_to_ohe(y_train, y_test)
        
        self._print_data_stats(X_train,y_train,X_test,y_test)
       
        return X_train, y_train_ohe, X_test, y_test_ohe, test_df

    def filter_single_trip(self, test_df, trip_prefix):
        return test_df[test_df[Column.TRIP_ID].str.contains(trip_prefix)].sort_index()

    def add_preds_to_df(self, test_df, y_pred_proba, hazard_2_best_conf, class_2_index):
        y_pred = utils.keep_high_conf_hazards(y_pred_proba, hazard_2_best_conf, class_2_index)
        test_df[Column.PRED] = pd.Series(y_pred, index=test_df.index)
        return test_df
