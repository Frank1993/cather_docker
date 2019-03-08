import pandas as pd
import numpy as np
import requests
import os
import apollo_python_common.map_geometry.geometry_utils as geometry_utils
from apollo_python_common.metadata.matching.match_column import MatchColumn as Column
from apollo_python_common.metadata.matching.match_params import MatchParams as MatchParams


class MetadataMatcher:
    
    OSM2_API_BASE_URL = 'https://orion-map.openstreetcam.org/'

    def __init__(self,reader):
        self.match_url = self.OSM2_API_BASE_URL + 'latest/match'
        self.reader = reader

    def __join_seq_df_with_match_df(self, meta_df, match_df):
        
        if match_df is None:
            meta_df[Column.FORWARD] = False
            meta_df[Column.WAY_ID] = -1
            meta_df[Column.FROM_NODE_ID] = -1 
            meta_df[Column.MATCHED_LAT] = -1 
            meta_df[Column.MATCHED_LON] = -1 
            meta_df[Column.COMPUTED_HEADING] = -1
            meta_df[Column.TO_NODE_ID] = -1 
            
            return meta_df
        
        joined_df = pd.merge(meta_df, match_df, how='outer', left_on=["timestamp"], right_on=[Column.DATETIME])
        joined_df = joined_df.drop([Column.DATETIME], axis=1)
        
        return joined_df

    def __get_heading_list(self, meta_df):
        rows = [r for _, r in meta_df.iterrows()]
        heading_list = []
            
        if len(rows) < 2:
            return [0] * len(rows)

        for pic1, pic2 in zip(rows, rows[1:]):
            heading = geometry_utils.compute_heading(float(pic1[Column.LAT]), float(pic1[Column.LON]),
                                                     float(pic2[Column.LAT]), float(pic2[Column.LON]))
            heading_list.append(heading)

        heading_list.append(heading_list[-1])

        return heading_list

    def __add_heading(self, df):
        heading_list = self.__get_heading_list(df)
        df[Column.HEADING] = pd.Series(heading_list, index=df.index)
        return df

    def __convert_2_match_server_format(self, meta_df):
        trip_probes = {MatchParams.PROBES: []}

        for _, row in meta_df.iterrows():
            trip_probes[MatchParams.PROBES].append({
                MatchParams.TIMESTAMP: row[Column.TIMESTAMP],
                MatchParams.LAT: row[Column.LAT],
                MatchParams.LON: row[Column.LON],
                MatchParams.PROPERTIES:
                    {MatchParams.ACCURACY: 1,
                     MatchParams.HEADING: row[Column.HEADING],
                     MatchParams.SPEED: 20
                     }
            })

        return trip_probes

    def __get_match_df(self, matched_sections_dict):        
        match_df = pd.DataFrame(matched_sections_dict).transpose()
        match_df = match_df.reset_index()
        match_df = match_df.rename({MatchParams.INDEX: Column.DATETIME,
                                    MatchParams.COMPUTED_HEADING: Column.COMPUTED_HEADING}, axis=1)
        match_df[Column.DATETIME] = match_df[Column.DATETIME].astype(np.int64)

        match_df[Column.WAY_ID] = match_df[MatchParams.ID].apply(lambda match_id: match_id[MatchParams.WAY_ID]).astype(int)
        match_df[Column.FROM_NODE_ID] = match_df[MatchParams.ID].apply(
            lambda match_id: match_id[MatchParams.FROM_NODE_ID]).astype(int)
        match_df[Column.TO_NODE_ID] = match_df[MatchParams.ID].apply(lambda match_id: match_id[MatchParams.TO_NODE_ID]).astype(int)

        match_df[Column.MATCHED_LAT] = match_df[MatchParams.MATCHED_POSITION].apply(lambda mp: mp[MatchParams.LAT])
        match_df[Column.MATCHED_LON] = match_df[MatchParams.MATCHED_POSITION].apply(lambda mp: mp[MatchParams.LON])

        for col in [Column.COMPUTED_HEADING, Column.OFFSET]:
            match_df[col] = match_df[col].astype(np.float64)

        match_df = match_df.drop([MatchParams.ID, MatchParams.MATCHED_POSITION], axis=1)
        
        return match_df

    def __convert_response_2_df(self, server_response):

        if MatchParams.TIME_BASED_MATCHED_SECTIONS not in server_response or len(server_response[MatchParams.TIME_BASED_MATCHED_SECTIONS]) == 0:
            print("No match data ")
            return None

        return self.__get_match_df(server_response[MatchParams.TIME_BASED_MATCHED_SECTIONS])

    def __make_request_to_server(self, meta_df):
        data_to_match = self.__convert_2_match_server_format(meta_df)
        return requests.post(self.match_url, json=data_to_match).json()

    def __filter_null_coordinates(self, meta_df):
        meta_df = meta_df[(meta_df[Column.LAT].notnull()) & (meta_df[Column.LON].notnull())]
        return meta_df.copy()

    def __read_df(self, metadata_path):
        meta_df = self.reader.read_metadata(metadata_path)
        return meta_df

    def __build_match_data(self, filtered_meta_df):
        filtered_meta_df = self.__add_heading(filtered_meta_df)
        server_response = self.__make_request_to_server(filtered_meta_df)
        match_df = self.__convert_response_2_df(server_response)
        return match_df

    def match_metadata(self, metadata_path):
        print(f"Start matching... {os.path.basename(metadata_path)}")
        meta_df = self.__read_df(metadata_path)
        filtered_meta_df = self.__filter_null_coordinates(meta_df)
        match_df = self.__build_match_data(filtered_meta_df)
        joined_df = self.__join_seq_df_with_match_df(meta_df, match_df)
        print(f"End matching... {os.path.basename(metadata_path)}")

        return joined_df
