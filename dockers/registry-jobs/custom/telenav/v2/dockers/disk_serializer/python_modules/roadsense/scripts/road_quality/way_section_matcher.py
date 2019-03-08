import requests
import pandas as pd
from xml.etree import ElementTree
from apollo_python_common.map_geometry.geometry_utils import compute_haversine_distance

class WaySection:
    
    def __init__(self,comp):
        self.components = [comp]
        
    def add_member(self,comp):
        self.components.append(comp)
    
    def get_head(self):
        return self.components[0]

    def get_tail(self):
        return self.components[-1]

    
class WaySectionMatcher:
    
    WAY_INFO_URL = "https://www.openstreetmap.org/api/0.6/way/{}/full"
    SECTION_MAX_LENGTH = 100
        
    NODE = "node"
    LAT = "lat"
    LON = "lon"
    NEXT_LAT = "next_lat"
    NEXT_LON = "next_lon"
    ID = "id"
    NODE_ID = "node_id"
    NEXT_NODE_ID = "next_node_id"
    WAY = "way"
    REF = "ref"
    ND = "nd"
    WS_ID = "ws_index"

    def __init__(self):
        self.way_id_2_nodes = {}
        self.way_id_2_sections = {}
        
    def _get_osm_way_info(self,way_id):
        resp = requests.get(self.WAY_INFO_URL.format(way_id))
        return ElementTree.fromstring(resp.text)
        
    def _get_node_df(self,osm_info_xml):
        node_df = pd.DataFrame(list(osm_info_xml.findall(self.NODE)),columns=[self.NODE]) 
        node_df[self.NODE_ID] = node_df[self.NODE].apply(lambda n: int(n.get(self.ID)))
        node_df[self.LAT] = node_df[self.NODE].apply(lambda n: float(n.get(self.LAT)))
        node_df[self.LON] = node_df[self.NODE].apply(lambda n: float(n.get(self.LON)))
        node_df = node_df.drop([self.NODE],axis=1)
        return node_df
    
    def _get_raw_way_df(self,osm_info_xml):
        raw_way_df = pd.DataFrame(osm_info_xml.find(self.WAY).findall(self.ND),columns=[self.ND])
        raw_way_df[self.NODE_ID] = raw_way_df[self.ND].apply(lambda nd: int(nd.get(self.REF)))
        raw_way_df = raw_way_df.drop([self.ND],axis=1)
        return raw_way_df #order is important for this df
        
    def _get_way_df(self, way_id):
        try:
            osm_info_xml = self._get_osm_way_info(way_id)
        except Exception as e:
            print(e)
            return None
            
        node_df = self._get_node_df(osm_info_xml)
        raw_way_df = self._get_raw_way_df(osm_info_xml)
        way_df = pd.merge(raw_way_df,node_df,how='inner',on=[self.NODE_ID])
        
        return way_df

    def _ensure_way_df_exists(self,way_id):
        if way_id not in self.way_id_2_nodes:
            way_df = self._get_way_df(way_id)
            self.way_id_2_nodes[way_id] = way_df
        return self.way_id_2_nodes[way_id]

    def _compute_way_sections(self, way_id):
            way_df = self._ensure_way_df_exists(way_id)

            if way_df is None:
                return None

            way_sections = []
            ws = WaySection(way_df.loc[0])
            initial_sections_rows = list(way_df.iterrows())

            for row_index,(_,row) in enumerate(initial_sections_rows[1:]): #ugly, but necessary
                node_lat, node_lon = row[self.LAT], row[self.LON]            
                dist_2_section_start = compute_haversine_distance(node_lon,node_lat,
                                                                  ws.get_head()[self.LON], 
                                                                  ws.get_head()[self.LAT])

                if len(ws.components) < 2:
                    ws.add_member(row)
                    continue

                if dist_2_section_start > self.SECTION_MAX_LENGTH or row_index == len(initial_sections_rows)-1:
                    ws.add_member(row)
                    way_sections.append(ws)
                    ws = WaySection(ws.get_tail())
                else:    
                    ws.add_member(row)

            if len(ws.components) != 1:
                way_sections.append(ws)

            return way_sections
    
    def ensure_way_sections_exists(self,way_id):
        if way_id not in self.way_id_2_sections:
            way_sections = self._compute_way_sections(way_id)
            self.way_id_2_sections[way_id] = way_sections

        return self.way_id_2_sections[way_id]

    def _is_in_rectangle(self,target_lat,target_lon,lat_1,lon_1,lat_2,lon_2):
        if lon_1 <= lon_2:
            left_lon,right_lon = lon_1,lon_2
        else:
            left_lon,right_lon = lon_2,lon_1
        
        if target_lon < left_lon or target_lon > right_lon:
            return False
        
        if lat_1 > lat_2:
            up_lat,down_lat = lat_1,lat_2
        else:
            up_lat,down_lat = lat_2,lat_1
            
        if target_lat > up_lat or target_lat < down_lat:
            return False
    
        return True

    def match_to_way_section(self,way_id, matched_lat, matched_lon):
        way_sections = self.ensure_way_sections_exists(way_id)
        
        if way_sections is None:
            print(f"{matched_lat},{matched_lon} couldn't be matched on {way_id}. No OSM data for {way_id}")
            return None
        
        matched_way_sections = []
        for i,ws in enumerate(way_sections):
            for start_comp,end_comp in zip(ws.components,ws.components[1:]):
                in_rect = self._is_in_rectangle(matched_lat,matched_lon,
                                                start_comp[self.LAT],start_comp[self.LON],
                                                end_comp[self.LAT],end_comp[self.LON])
                
                if in_rect:
                    matched_way_sections.append(ws)
                                      
                
        if len(matched_way_sections) == 0:
            return None

        if len(matched_way_sections) > 1:
            return None
    
        return matched_way_sections[0]