import pandas as pd
from apollo_python_common.metadata.reader.abstract_metadata_reader import AbstractMetadataReader


class MetadataV2Reader(AbstractMetadataReader):
    name_convert_dict = {
        "videoIndex": "vIndex",
        "frameIndex": "tFIndex",
        "latitude": "lat",
        "longitude": "lon",
        "horizontalAccuracy": "h_accu",
        "GPSspeed": "GPSs",
        "compass": "comp",
        "OBDspeed": "OBDs",
        "altitude": "elv",
        "accelerationX": "accX",
        "accelerationY": "accY",
        "accelerationZ": "accZ",
        "verticalAccuracy": "v_accu",
        "gravityX": "gX",
        "gravityY": "gY",
        "gravityZ": "gZ"
    }

    metadata_format_dict = {
        "f": ["videoIndex", "frameIndex", "gpsTimestamp", "latitude", "longitude",
              "horizontalAccuracy", "GPSspeed", "compasTimestamp", "compass", "obdTimestamp", "OBDspeed"],
        "g": ["latitude", "longitude", "altitude", "horizontalAccuracy", "verticalAccuracy", "GPSspeed"],
        "m": ["yaw", "pitch", "roll", "accelerationX", "accelerationY", "accelerationZ", "gravityX", "gravityY",
              "gravityZ"],
        "d": ["platformName", "osRawName", "osVersion", "deviceRawName", "appVersion", "appBuildNumber",
              "recordingType"],
    }

    feature_functions = ["f", "g", "m"]

    def __read_file_lines(self, path):
        file = open(path, "r")
        lines = file.readlines()
        lines = [l.replace("\n", "") for l in lines]

        return lines

    def __filter_body_lines(self, lines):
        index_of_body = [i for i, l in enumerate(lines) if l == "BODY"][0]
        return lines[index_of_body + 1:-1]

    def __empty_row(self):
        return {name: None for name in self.SENSOR_NAMES}

    def __filled_row(self, timestamp, function, arg_values):
        row = self.__empty_row()
        arg_names = self.metadata_format_dict[function]
        arg_dict = dict(zip(arg_names, arg_values))

        for arg_name, arg_value in arg_dict.items():
            col_name = self.name_convert_dict[arg_name] if arg_name in self.name_convert_dict else arg_name
            row[col_name] = arg_value

        row["timestamp"] = timestamp
        return row

    def __get_row_components(self, line):
        splits = line.split(":")
        timestamp, function, arg_values = splits[0], splits[1], splits[2].split(";")
        return timestamp, function, arg_values

    def __file_line_2_row(self, line):
        timestamp, function, arg_values = self.__get_row_components(line)

        if function in self.feature_functions:
            return self.__filled_row(timestamp, function, arg_values)

        return self.__empty_row()

    def read_metadata(self, metadata_path):
        lines = self.__read_file_lines(metadata_path)
        lines = self.__filter_body_lines(lines)
        row_list = [self.__file_line_2_row(line) for line in lines]
        return pd.DataFrame(row_list).dropna(axis=0, how='all')

    def get_device_name(self, metadata_path):
        print(metadata_path)
        lines = self.__read_file_lines(metadata_path)
        lines = self.__filter_body_lines(lines)
        timestamp, function, arg_values = self.__get_row_components(lines[0])

        arg_names = self.metadata_format_dict[function]
        arg_dict = dict(zip(arg_names, arg_values))
        return arg_dict["deviceRawName"]
