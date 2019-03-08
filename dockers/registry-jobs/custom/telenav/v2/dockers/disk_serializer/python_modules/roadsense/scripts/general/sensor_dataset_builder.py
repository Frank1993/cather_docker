import os
import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, Normalizer
from tqdm import tqdm

from apollo_python_common.metadata.reader.metadata_reader import MetadataReader
from apollo_python_common.metadata.matching.meta_matcher import MetadataMatcher
import apollo_python_common.io_utils as io_utils

import roadsense.scripts.general.signal_processing as sp
import roadsense.scripts.general.utils as utils
from roadsense.scripts.general.config import ConfigParams as cp, HazardType as HazardType, Column
from roadsense.scripts.road_quality.way_section_matcher import WaySectionMatcher

class SensorDatasetBuilder:
    BATCH_LENGTH = 1000
    SENSOR_NAMES = ['timestamp', 'lon', 'lat', 'elv', 'h_accu', 'GPSs', 'yaw', 'pitch', 'roll',
                    'accX', 'accY', 'accZ', 'pres', 'comp', 'vIndex', 'tFIndex', 'gX', 'gY', 'gZ', 'OBDs', 'v_accu']

    START_CROP_BUFFER_PERC = 0.05

    def __init__(self, config):
        self.config = config
        self.reader = MetadataReader()
        self.matcher = MetadataMatcher(self.reader)
        self.ws_matcher = WaySectionMatcher()
        
    def _read_sensor_df(self, sensor_path):
        if self.config[cp.ADD_MATCH_DATA]:
            df = self.matcher.match_metadata(sensor_path)
        else:
            df = self.reader.read_metadata(sensor_path)
            
        df[Column.IMAGE_INDEX] = df[Column.IMAGE_INDEX].ffill()
        return df

    def _filter_columns(self, df, cols):
        return df[[Column.TIMESTAMP] + cols]

    def _add_datetime(self, sensor_df):
        sensor_df.loc[:, Column.DATETIME] = pd.to_datetime(sensor_df[Column.TIMESTAMP], unit='ms')
        return sensor_df.set_index(Column.DATETIME).drop(Column.TIMESTAMP, axis=1)

    def _resample_df(self, df):
        freq = self.config[cp.FREQUENCY]
        return df.resample(f"{freq}L").mean().ffill().bfill()

    def _crop_start(self, sensor_df):
        return sensor_df.sort_index()[int(len(sensor_df) * self.START_CROP_BUFFER_PERC):]

    def _add_prefix_to_cols(self, df, prefix):
        df.columns = [f"{col}_{prefix}" for col in df.columns]
        return df, list(df.columns)

    def _add_derived_features(self, df, cols_of_interest):
        rolling_df = df[cols_of_interest].rolling(self.config[cp.DERIVED_WINDOW_SIZE], min_periods=3)
        mean_df, mean_df_col_names = self._add_prefix_to_cols(rolling_df.mean(), "mean")
        max_df, max_df_col_names = self._add_prefix_to_cols(rolling_df.max(), "max")
        min_df, min_df_col_names = self._add_prefix_to_cols(rolling_df.min(), "min")
        std_df, std_dev_col_name = self._add_prefix_to_cols(rolling_df.std(), "std")

        all_features_names = mean_df_col_names + max_df_col_names + min_df_col_names + std_dev_col_name

        return pd.concat([df, mean_df, max_df, min_df, std_df], axis=1).dropna(), all_features_names

    def _merge_sensor_with_hazard_by_timestamp(self, sensor_df, hazard_df):
        min_date, max_date = sensor_df.index[0], sensor_df.index[-1]

        merged_df = hazard_df.loc[min_date:max_date, :].join(sensor_df, how='outer')
        merged_df[Column.RAW_HAZARD] = merged_df[Column.RAW_HAZARD].fillna(value=HazardType.CLEAR)
        merged_df[Column.HAZARD_LAT] = merged_df[Column.HAZARD_LAT].fillna(value=-1)
        merged_df[Column.HAZARD_LON] = merged_df[Column.HAZARD_LON].fillna(value=-1)

        return merged_df

    def _normalize_data(self, df, cols_of_interest):

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

    def _change_numeric_type(self, sensor_df, feature_cols, target_type):
        sensor_df[feature_cols] = sensor_df[feature_cols].astype(target_type)
        return sensor_df

    def _add_id_col(self, df, sensor_path):
        sensor_df_id = utils.get_sensor_df_id(sensor_path,self.reader)
        df[Column.TRIP_ID] = sensor_df_id
        return df

    def _series_to_supervised(self, orig_df, columns_of_interest=None):

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

    def _collect_window_values(self, sensor_df, col):

        past_steps = future_steps = self.config[cp.STEPS] // 2

        past_range = range(past_steps, -1, -1)
        future_range = range(1, future_steps + 1)

        past_cols = [f"{col}_-{i}" for i in past_range]
        future_cols = [f"{col}_+{i}" for i in future_range]
        window_cols = past_cols + future_cols

        sensor_df.loc[:, f"{col}_window"] = pd.Series(sensor_df[window_cols].values.tolist(), index=sensor_df.index)
        sensor_df = sensor_df.drop(window_cols, axis=1)

        return sensor_df

    def _get_window_features_df(self, sensor_df, cols_of_interest):

        sensor_df = self._series_to_supervised(sensor_df, cols_of_interest)
        for col in cols_of_interest:
            sensor_df = self._collect_window_values(sensor_df, col)

        return sensor_df

    def _compute_features_stats(self, x):
        functions = [np.mean, np.min, np.max, np.std, np.var, kurtosis, skew]
        x = np.stack([f(x.astype(np.float64), axis=1) for f in functions], axis=1).astype(np.float32)
        x = x.reshape(x.shape[0], -1)
        return x

    def _add_merged_features_col(self, df, cols_of_interest):
        window_col_names = [f"{col}_window" for col in cols_of_interest]
        df.loc[:, Column.ALL_FEATURES] = df[window_col_names].values.tolist()
        df.loc[:, Column.ALL_FEATURES] = df.loc[:, Column.ALL_FEATURES].apply(np.array)
        df.loc[:, Column.ALL_FEATURES] = df.loc[:, Column.ALL_FEATURES].apply(self._compute_features_stats)
        return df.drop(cols_of_interest, axis=1)

    def _get_hazard_type(self, hazard_list):
        buffer = self.config[cp.HAZARD_BUFFER_STEPS]
        short_hazard_list = hazard_list[buffer:len(hazard_list) - buffer]

        hazards_by_impact = [HazardType.UNPAVED_ROAD,
                             HazardType.BUMPY_ROAD,
                             HazardType.SPEED_BUMP,
                             HazardType.BIG_POTHOLE,
                             HazardType.SMALL_POTHOLE,
                             HazardType.SEWER_HOLE]

        for hazard in hazards_by_impact:
            if hazard in short_hazard_list:
                return hazard

        return HazardType.CLEAR

    def _update_hazard_column(self, df):
        df.loc[:, Column.HAZARD] = df.loc[:, "{}_window".format(Column.RAW_HAZARD)].apply(self._get_hazard_type)
        return df

    def _remove_window_features(self, sensor_df, cols_of_interest):
        return sensor_df.drop([f"{c}_window" for c in cols_of_interest], axis=1)

    def _save_sensor_df_to_disk(self, df, drive_folder, index):
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
        sensor_df = self._get_window_features_df(sensor_df, feature_cols + [Column.RAW_HAZARD])
        sensor_df = self._add_merged_features_col(sensor_df, feature_cols)
        sensor_df = self._update_hazard_column(sensor_df)
        sensor_df = self._remove_window_features(sensor_df, feature_cols + [Column.RAW_HAZARD])
        self._save_sensor_df_to_disk(sensor_df, drive_folder, index)

        return sensor_df

    def _add_window_features(self, sensor_df, feature_cols, drive_folder):

        nr_batches = max(1, len(sensor_df) // self.BATCH_LENGTH + 1)
        nr_threads = min(40, nr_batches)

        print(f"Nr of data batches: {nr_batches}")
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

    def _hazard_start_stop_clear(self,h):
        return HazardType.CLEAR if h in set([HazardType.BUMPY_ROAD_START,HazardType.BUMPY_ROAD_END, 
                                  HazardType.UNPAVED_ROAD_START,HazardType.UNPAVED_ROAD_END]) else h                


    def _add_is_paved_info(self,sensor_df):
        br_start_times = sensor_df[sensor_df[Column.RAW_HAZARD] == HazardType.BUMPY_ROAD_START].index.tolist()
        br_end_times = sensor_df[sensor_df[Column.RAW_HAZARD] == HazardType.BUMPY_ROAD_END].index.tolist()
        assert(len(br_start_times) == len(br_end_times))
        
        unpaved_times = sensor_df[sensor_df[Column.RAW_HAZARD] == HazardType.UNPAVED_ROAD].index.tolist()
        
        for br_start_time, br_end_time in zip(br_start_times,br_end_times):
            for unpaved_time in unpaved_times:
                if br_start_time <= unpaved_time <= br_end_time:
                    sensor_df.loc[br_start_time,Column.RAW_HAZARD] = HazardType.UNPAVED_ROAD_START
                    sensor_df.loc[br_end_time,Column.RAW_HAZARD] = HazardType.UNPAVED_ROAD_END

        for unpaved_time in unpaved_times:
            sensor_df = sensor_df.drop(unpaved_time)
        
        return sensor_df
        
    def _fill_all_windows_with_continous_hazard(self, sensor_df, start_hazard, end_hazard, fill_hazard):
        start_times = sensor_df[sensor_df[Column.RAW_HAZARD] == start_hazard].index.tolist()
        end_times = sensor_df[sensor_df[Column.RAW_HAZARD] == end_hazard].index.tolist()
        assert(len(start_times) == len(end_times))

        for start_time,end_time in zip(start_times,end_times):
            assert(end_time >= start_time)
            sensor_df.loc[start_time:end_time,Column.RAW_HAZARD] = fill_hazard

        return sensor_df
    
    def _remove_start_end_hazard_events(self, sensor_df):
        sensor_df[Column.RAW_HAZARD] = sensor_df[Column.RAW_HAZARD].apply(self._hazard_start_stop_clear)
        return sensor_df
    
    def _handle_bumpy_road_events(self, sensor_df):
        sensor_df = self._add_is_paved_info(sensor_df)
        sensor_df = self._fill_all_windows_with_continous_hazard(sensor_df, HazardType.BUMPY_ROAD_START,
                                                                  HazardType.BUMPY_ROAD_END, HazardType.BUMPY_ROAD)
        sensor_df = self._fill_all_windows_with_continous_hazard(sensor_df, HazardType.UNPAVED_ROAD_START,
                                                                  HazardType.UNPAVED_ROAD_END, HazardType.UNPAVED_ROAD)
        
        sensor_df = self._remove_start_end_hazard_events(sensor_df)    

        return sensor_df

    def _get_non_feature_cols(self):
        non_feature_cols =  [Column.LAT, Column.LON, Column.IMAGE_INDEX]
        
        if self.config[cp.ADD_MATCH_DATA]:
            non_feature_cols += [Column.FORWARD,Column.WAY_ID,Column.FROM_NODE_ID,
                                 Column.TO_NODE_ID,Column.MATCHED_LAT,Column.MATCHED_LON, 
                                 Column.COMPUTED_HEADING]        
        
        return non_feature_cols
    
    def _add_custom_way_section_match_data(self,sensor_df):
        print("Started adding custom way-sections")
        matched_ws_list = []
        for _, row in tqdm(list(sensor_df.iterrows())):
            matched_ws = self.ws_matcher.match_to_way_section(int(row[Column.WAY_ID]),
                                                              row[Column.MATCHED_LAT],
                                                              row[Column.MATCHED_LON])
            matched_ws_list.append(matched_ws)

        sensor_df[Column.SECTION_FROM_NODE_ID] = [int(ws.get_head()[Column.NODE_ID]) if ws is not None else -1 
                                                  for ws in matched_ws_list]
        
        sensor_df[Column.SECTION_FROM_NODE_ID_LAT] = [ws.get_head()[Column.LAT] if ws is not None else -1 
                                                      for ws in matched_ws_list]

        sensor_df[Column.SECTION_FROM_NODE_ID_LON] = [ws.get_head()[Column.LON] if ws is not None else -1 
                                                      for ws in matched_ws_list]

        sensor_df[Column.SECTION_TO_NODE_ID] = [int(ws.get_tail()[Column.NODE_ID]) if ws is not None else -1 
                                                for ws in matched_ws_list]

        sensor_df[Column.SECTION_TO_NODE_ID_LAT] = [ws.get_tail()[Column.LAT] if ws is not None else -1 
                                                    for ws in matched_ws_list]

        sensor_df[Column.SECTION_TO_NODE_ID_LON] = [ws.get_tail()[Column.LON] if ws is not None else -1 
                                                    for ws in matched_ws_list]

        print("Ended adding custom way-sections")
        return sensor_df
        
        
    def construct_trip_sensor_df(self, sensor_path, hazard_df, drive_folder):
        feature_cols = self.config[cp.FEATURES].copy()
        non_feature_cols = self._get_non_feature_cols()
        print(f"Starting... {os.path.basename(sensor_path)}")

        sensor_df = self._read_sensor_df(sensor_path)
        sensor_df = self._filter_columns(sensor_df, feature_cols + non_feature_cols)
        sensor_df = self._add_datetime(sensor_df)
        sensor_df = self._change_numeric_type(sensor_df, 
                                              feature_cols + [Column.LAT, Column.LON, Column.IMAGE_INDEX], np.float32)        
        
        sensor_df = self._resample_df(sensor_df)   
        
        if self.config[cp.WITH_CUSTOM_WAY_SECTIONS]:   
            sensor_df = self._change_numeric_type(sensor_df, [Column.IMAGE_INDEX,
                                                              Column.WAY_ID,Column.FROM_NODE_ID,Column.TO_NODE_ID],int)
            sensor_df = self._add_custom_way_section_match_data(sensor_df)

        if self.config[cp.CROP_START]:
            sensor_df = self._crop_start(sensor_df)

        sensor_df, proj_features_col_names = sp.add_projection_features(sensor_df, min_nr_obs_between_peeks=10)
        feature_cols += proj_features_col_names

        sensor_df, derived_col_names = self._add_derived_features(sensor_df, feature_cols)
        feature_cols += derived_col_names

        if len(sensor_df) == 0:
            print("Empty sensor df")
            return

        sensor_df = self._merge_sensor_with_hazard_by_timestamp(sensor_df, hazard_df)
        sensor_df = self._handle_bumpy_road_events(sensor_df)
        sensor_df = self._normalize_data(sensor_df, feature_cols)

        sensor_df = self._add_id_col(sensor_df, sensor_path)

        self._add_window_features(sensor_df, feature_cols, drive_folder)