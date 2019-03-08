from apollo_python_common.metadata.reader.abstract_metadata_reader import AbstractMetadataReader
from apollo_python_common.metadata.reader.metadata_v1_reader import MetadataV1Reader
from apollo_python_common.metadata.reader.metadata_v2_reader import MetadataV2Reader


class MetadataReader(AbstractMetadataReader):
    V2_FIRST_LINE = "METADATA:2.0"
    TIMESTAMP_COL = "timestamp"

    def __init__(self):
        self.v1_reader = MetadataV1Reader()  # v1 reader works only for metadata versions >= 1.1
        self.v2_reader = MetadataV2Reader()

    def _get_reader_for_metadata_version(self, metadata_path):
        file = open(metadata_path, "r")
        first_line = file.readline().replace("\n", "")

        if first_line == self.V2_FIRST_LINE:
            return self.v2_reader

        return self.v1_reader

    def _format_timestamp(self, df):
        df = df[df[self.TIMESTAMP_COL].notnull()].copy()
        df[self.TIMESTAMP_COL] = df[self.TIMESTAMP_COL].apply(
            lambda ts: int(str(ts).replace(".", "")[:13].ljust(13, '0')))
        return df

    def read_metadata(self, metadata_path):
        df = self._get_reader_for_metadata_version(metadata_path).read_metadata(metadata_path)
        df = self._format_timestamp(df)
        return df

    def get_device_name(self, metadata_path):
        return self._get_reader_for_metadata_version(metadata_path).get_device_name(metadata_path)
