import json
import os
from glob import glob

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

from roadsense.scripts.config import ConfigParams as cp, Column, FolderName as fn


class HazardDatasetBuilder:
    TAGS_KEY = "tags"
    TYPE_KEY = "type"

    def __init__(self, config):
        self.config = config

    def __handle_location_cols(self, df):
        df.loc[:, Column.HAZARD_LAT] = df.loc[:, Column.LOCATION].apply(lambda l: l.split(",")[0])
        df.loc[:, Column.HAZARD_LON] = df.loc[:, Column.LOCATION].apply(lambda l: l.split(",")[1])
        return df.drop(Column.LOCATION, axis=1)

    def __read_hazard_df(self, hazard_folder, specific_hazards=None):
        json_path = glob(hazard_folder + "/*")[0]

        with open(json_path) as json_data_file:
            hazard_dict = json.load(json_data_file)

        hazard_df = pd.DataFrame(hazard_dict[self.TAGS_KEY])
        hazard_df = hazard_df.rename(columns={self.TYPE_KEY: Column.RAW_HAZARD})

        if specific_hazards is not None:
            hazard_df = hazard_df[hazard_df[Column.RAW_HAZARD].isin(specific_hazards)]

        return hazard_df

    def __add_datetime(self, hazard_df):
        hazard_df.loc[:, Column.DATETIME] = pd.to_datetime(hazard_df[Column.TIMESTAMP], unit='ms')
        hazard_df.loc[:, Column.DATETIME] = hazard_df[Column.DATETIME].dt.round("{}L".format(self.config[cp.FREQUENCY]))
        return hazard_df.set_index(Column.DATETIME).drop([Column.TIMESTAMP], axis=1)

    def get_hazard_df(self, drive_folder):
        hazard_folder = os.path.join(drive_folder, fn.HAZARD_DATA)

        if len(os.listdir(hazard_folder)) == 0:
            print("Hazards not founds. Assuming no hazzards on trip")
            return pd.DataFrame(columns=[Column.RAW_HAZARD, Column.HAZARD_LAT, Column.HAZARD_LON])

        hazard_df = self.__read_hazard_df(hazard_folder, self.config[cp.SPECIFIC_HAZARDS])
        hazard_df = self.__handle_location_cols(hazard_df)
        hazard_df = self.__add_datetime(hazard_df)
        return hazard_df
