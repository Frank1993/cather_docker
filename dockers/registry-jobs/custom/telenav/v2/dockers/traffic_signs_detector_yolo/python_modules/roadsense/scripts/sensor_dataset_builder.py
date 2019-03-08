import os
from functools import partial
from multiprocessing import Pool

import apollo_python_common.io_utils as io_utils
import numpy as np
import pandas as pd
import roadsense.scripts.signal_processing as sp
import roadsense.scripts.utils as utils
from roadsense.scripts.config import ConfigParams as cp, HazardType as HazardType, Column
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, Normalizer
from tqdm import tqdm

tqdm.pandas()


class SensorDatasetBuilder:
    BATCH_LENGTH = 1000
    SENSOR_NAMES = ['timestamp', 'lon', 'lat', 'elv', 'h_accu', 'GPSs', 'yaw', 'pitch', 'roll',
                    'accX', 'accY', 'accZ', 'pres', 'comp', 'vIndex', 'tFIndex', 'gX', 'gY', 'gZ', 'OBDs', 'v_accu']

    START_CROP_BUFFER_PERC = 0.05

    def __init__(self, config):
        self.config = config

    def __read_sensor_df(self, sensor_path):
        df = pd.read_csv(sensor_path, sep=";", skiprows=1, names=self.SENSOR_NAMES, index_col=False)
        df[Column.IMAGE_INDEX] = df[Column.IMAGE_INDEX].ffill()
        return df

    def __filter_columns(self, df, cols):
        return df[[Column.TIMESTAMP] + cols]

    def __format_sensor_timestamp(self, df):
        df.loc[:, Column.TIMESTAMP] = df.loc[:, Column.TIMESTAMP].apply(lambda ts:
                                                                            int(str(ts).replace(".", "")[:13].ljust(13,
                                                                                                                    '0')))
        return df

    def __add_datetime(self, sensor_df):
        sensor_df.loc[:, Column.DATETIME] = pd.to_datetime(sensor_df[Column.TIMESTAMP], unit='ms')
        return sensor_df.set_index(Column.DATETIME).drop(Column.TIMESTAMP, axis=1)

    def __resample_df(self, df):
        freq = self.config[cp.FREQUENCY]
        return df.resample(f"{freq}L").mean().ffill().bfill()

    def __crop_start(self, sensor_df):
        return sensor_df.sort_index()[int(len(sensor_df) * self.START_CROP_BUFFER_PERC):]

    def __add_prefix_to_cols(self, df, prefix):
        df.columns = [f"{col}_{prefix}" for col in df.columns]
        return df, list(df.columns)

    def __add_derived_features(self, df, cols_of_interest):
        rolling_df = df[cols_of_interest].rolling(self.config[cp.DERIVED_WINDOW_SIZE], min_periods=3)
        mean_df, mean_df_col_names = self.__add_prefix_to_cols(rolling_df.mean(), "mean")
        max_df, max_df_col_names = self.__add_prefix_to_cols(rolling_df.max(), "max")
        min_df, min_df_col_names = self.__add_prefix_to_cols(rolling_df.min(), "min")
        std_df, std_dev_col_name = self.__add_prefix_to_cols(rolling_df.std(), "std")

        all_features_names = mean_df_col_names + max_df_col_names + min_df_col_names + std_dev_col_name

        return pd.concat([df, mean_df, max_df, min_df, std_df], axis=1).dropna(), all_features_names

    def __merge_sensor_with_hazard_by_timestamp(self, sensor_df, hazard_df):
        min_date, max_date = sensor_df.index[0], sensor_df.index[-1]

        merged_df = hazard_df.loc[min_date:max_date, :].join(sensor_df, how='outer')
        merged_df[Column.RAW_HAZARD] = merged_df[Column.RAW_HAZARD].fillna(value=HazardType.CLEAR)
        merged_df[Column.HAZARD_LAT] = merged_df[Column.HAZARD_LAT].fillna(value=-1)
        merged_df[Column.HAZARD_LON] = merged_df[Column.HAZARD_LON].fillna(value=-1)

        return merged_df

    def __normalize_data(self, df, cols_of_interest):

        scaler_type = self.config[cp.SCALER_TYPE]
        if scaler_type == "na":
            return df

        if scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        elif scaler_type == "maxabs":
            scaler = MaxAbsScaler()
        elif scaler_type == "normalizer":
            scaler = Normalizer()
        else:
            raise Exception("Invalid scaler_type. Must be [na, minmax, standard, robust, maxabs,normalizer]")

        df[cols_of_interest] = scaler.fit_transform(df[cols_of_interest])

        return df

    def __change_numeric_type(self, sensor_df, feature_cols):
        sensor_df[feature_cols] = sensor_df[feature_cols].astype(np.float32)
        return sensor_df

    def __add_id_col(self, df, sensor_path):
        sensor_df_id = utils.get_sensor_df_id(sensor_path)
        df[Column.TRIP_ID] = sensor_df_id
        return df

    def __series_to_supervised(self, orig_df, columns_of_interest=None):

        past_steps = future_steps = self.config[cp.STEPS] // 2

        if columns_of_interest is None:
            columns_of_interest = list(orig_df.columns)

        df = orig_df[columns_of_interest]
        cols = list()
        names = list()

        for i in range(past_steps, -1, -1):
            cols.append(df.shift(i))
            names += [('%s_-%d' % (col_name, i)) for col_name in columns_of_interest]

        for i in range(1, future_steps + 1):
            cols.append(df.shift(-i))
            names += [('%s_+%d' % (col_name, i)) for col_name in columns_of_interest]

        names += [col_name for col_name in orig_df]
        cols.append(orig_df.shift(0))

        agg = pd.concat(cols, axis=1)
        agg.columns = names

        agg = agg.dropna()
        return agg

    def __collect_window_values(self, sensor_df, col):

        past_steps = future_steps = self.config[cp.STEPS] // 2

        past_range = range(past_steps, -1, -1)
        future_range = range(1, future_steps + 1)

        past_cols = [f"{col}_-{i}" for i in past_range]
        future_cols = [f"{col}_+{i}" for i in future_range]
        window_cols = past_cols + future_cols

        sensor_df.loc[:, f"{col}_window"] = pd.Series(sensor_df[window_cols].values.tolist(), index=sensor_df.index)
        sensor_df = sensor_df.drop(window_cols, axis=1)

        return sensor_df

    def __get_window_features_df(self, sensor_df, cols_of_interest):

        sensor_df = self.__series_to_supervised(sensor_df, cols_of_interest)
        for col in cols_of_interest:
            sensor_df = self.__collect_window_values(sensor_df, col)

        return sensor_df

    def __compute_features_stats(self, x):
        functions = [np.mean, np.min, np.max, np.std, np.var, kurtosis, skew]
        x = np.stack([f(x.astype(np.float64), axis=1) for f in functions], axis=1).astype(np.float32)
        x = x.reshape(x.shape[0], -1)
        return x

    def __add_merged_features_col(self, df, cols_of_interest):
        window_col_names = [f"{col}_window" for col in cols_of_interest]
        df.loc[:, Column.ALL_FEATURES] = df[window_col_names].values.tolist()
        df.loc[:, Column.ALL_FEATURES] = df.loc[:, Column.ALL_FEATURES].apply(np.array)
        df.loc[:, Column.ALL_FEATURES] = df.loc[:, Column.ALL_FEATURES].apply(self.__compute_features_stats)
        return df.drop(cols_of_interest, axis=1)

    def __get_hazard_type(self, hazard_list):
        buffer = self.config[cp.HAZARD_BUFFER_STEPS]
        short_hazard_list = hazard_list[buffer:len(hazard_list) - buffer]

        hazards_by_impact = [HazardType.SPEED_BUMP,
                             HazardType.BIG_POTHOLE,
                             HazardType.SMALL_POTHOLE,
                             HazardType.SEWER_HOLE]

        for hazard in hazards_by_impact:
            if hazard in short_hazard_list:
                return hazard

        return HazardType.CLEAR

    def __update_hazard_column(self, df):
        df.loc[:, Column.HAZARD] = df.loc[:, "{}_window".format(Column.RAW_HAZARD)].apply(self.__get_hazard_type)
        return df

    def __remove_window_features(self, sensor_df, cols_of_interest):
        return sensor_df.drop([f"{c}_window" for c in cols_of_interest], axis=1)

    def __save_sensor_df_to_disk(self, df, drive_folder, index):
        if len(df) == 0:
            return

        output_path = os.path.join(self.config[cp.DATASET_BASE_PATH], \
                                   utils.get_dataset_folder_name(self.config), \
                                   os.path.basename(drive_folder))

        io_utils.create_folder(output_path)
        df_id = df[Column.TRIP_ID].tolist()[0]
        df_name = f"{df_id}_{index}.p"

        df.to_pickle(os.path.join(output_path, df_name))

    def construct_window_features_df(self, sensor_df_2_index, feature_cols, drive_folder):
        sensor_df, index = sensor_df_2_index
        sensor_df = self.__get_window_features_df(sensor_df, feature_cols + [Column.RAW_HAZARD])
        sensor_df = self.__add_merged_features_col(sensor_df, feature_cols)
        sensor_df = self.__update_hazard_column(sensor_df)
        sensor_df = self.__remove_window_features(sensor_df, feature_cols + [Column.RAW_HAZARD])
        self.__save_sensor_df_to_disk(sensor_df, drive_folder, index)

        return sensor_df

    def __add_window_features(self, sensor_df, feature_cols, drive_folder):

        nr_batches = max(1, len(sensor_df) // self.BATCH_LENGTH + 1)
        nr_threads = min(40, nr_batches)

        print(f"Nr batches: {nr_batches}")
        indexes = list(range(nr_batches))
        sensor_df_list = np.array_split(sensor_df, nr_batches)
        pool = Pool(nr_threads)
        sdf_list = pool.map(partial(self.construct_window_features_df,
                                    feature_cols=feature_cols,
                                    drive_folder=drive_folder
                                    ),
                            zip(sensor_df_list, indexes))

        pool.close()

        return sdf_list

    def construct_trip_sensor_df(self, sensor_path, hazard_df, drive_folder):
        feature_cols = self.config[cp.FEATURES].copy()

        print("Start: {}".format(os.path.basename(sensor_path)))

        sensor_df = self.__read_sensor_df(sensor_path)
        sensor_df = self.__filter_columns(sensor_df, feature_cols + [Column.LAT, Column.LON, Column.IMAGE_INDEX])
        sensor_df = self.__format_sensor_timestamp(sensor_df)
        sensor_df = self.__add_datetime(sensor_df)
        sensor_df = self.__resample_df(sensor_df)
        sensor_df = self.__crop_start(sensor_df)

        sensor_df, proj_features_col_names = sp.add_projection_features(sensor_df, min_nr_obs_between_peeks=10)
        feature_cols += proj_features_col_names

        sensor_df, derived_col_names = self.__add_derived_features(sensor_df, feature_cols)
        feature_cols += derived_col_names

        if len(sensor_df) == 0:
            print("Empty sensor df")
            return

        sensor_df = self.__merge_sensor_with_hazard_by_timestamp(sensor_df, hazard_df)
        sensor_df = self.__normalize_data(sensor_df, feature_cols)

        sensor_df = self.__change_numeric_type(sensor_df, feature_cols)
        sensor_df = self.__add_id_col(sensor_df, sensor_path)

        self.__add_window_features(sensor_df, feature_cols, drive_folder)
