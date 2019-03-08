import pandas as pd
from apollo_python_common.metadata.reader.abstract_metadata_reader import AbstractMetadataReader


class MetadataV1Reader(AbstractMetadataReader):

    def read_metadata(self, metadata_path):
        return pd.read_csv(metadata_path, sep=";", skiprows=1, names=self.SENSOR_NAMES, index_col=False)[
               :-1]  # last line contains only "DONE" string

    def get_device_name(self, metadata_path):
        with open(metadata_path) as file:
            phone_type = "-".join(file.readline().split(";")[0].split(" "))
        return phone_type
