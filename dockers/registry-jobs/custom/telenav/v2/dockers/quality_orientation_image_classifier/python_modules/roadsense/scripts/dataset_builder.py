import argparse
import logging
import os
from glob import glob

from tqdm import tqdm

tqdm.pandas()

import roadsense.scripts.utils as utils
from roadsense.scripts.config import ConfigParams as cp, FolderName as fn
from roadsense.scripts.hazard_dataset_builder import HazardDatasetBuilder
from roadsense.scripts.sensor_dataset_builder import SensorDatasetBuilder

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util


class DatasetBuilder:

    def __init__(self, config):
        self.config = config
        self.hazard_builder = HazardDatasetBuilder(config)
        self.sensor_builder = SensorDatasetBuilder(config)

    def save_config_to_disk(self):
        config_save_path = os.path.join(self.config[cp.DATASET_BASE_PATH],
                                        utils.get_dataset_folder_name(self.config), fn.CONFIG, "config.json")
        io_utils.json_dump(self.config, config_save_path)

    def read_sensor_paths(self, drive_folder):
        sensor_folder = os.path.join(drive_folder, fn.OSC_SENSOR_DATA)
        return glob(sensor_folder + "/*")

    def apply_filters(self, sensor_paths):
        sensor_paths = [x for x in sensor_paths if os.path.basename(x) not in self.config[cp.BLACKLIST]]

        if self.config[cp.PHONE_NAME] != "all":
            sensor_paths = [x for x in sensor_paths if self.config[cp.PHONE_NAME] in utils.get_sensor_df_id(x).lower()]

        return sensor_paths

    def construct_dataset(self):

        for drive_folder in self.config[cp.DRIVE_FOLDERS]:
            hazard_df = self.hazard_builder.get_hazard_df(drive_folder)
            sensor_paths = self.read_sensor_paths(drive_folder)
            sensor_paths = self.apply_filters(sensor_paths)
            print(f"Nr sensor paths: {len(sensor_paths)}")

            for sensor_path in sensor_paths:
                self.sensor_builder.construct_trip_sensor_df(sensor_path, hazard_df, drive_folder)

        self.save_config_to_disk()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config_json", help="path to config json", type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':

    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    config = dict(io_utils.json_load(args.config_json))

    try:
        DatasetBuilder(config).construct_dataset()
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
