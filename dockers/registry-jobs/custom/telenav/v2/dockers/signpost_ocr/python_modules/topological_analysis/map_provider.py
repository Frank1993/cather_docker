import requests
from constants import OSM2_API_BASE_URL


class MapProvider:
    def __init__(self):
        self.__latest_map = self.get_latest_map()

    def get_latest_map(self):
        url_maps = OSM2_API_BASE_URL + 'maps'
        self.__latest_map = requests.get(url_maps).json()[0]
        return self.__latest_map

    def get_near_ways(self, gps_point, radius=500):
        near_url = OSM2_API_BASE_URL + '{map}/near/{lat}/{lon}/{radius}'
        near = requests.get(
            near_url.format(map=self.__latest_map, lat=gps_point.latitude, lon=gps_point.longitude, radius=radius))
        return near.json()

    def get_way_info(self, way_id, from_node_id, to_node_id):
        way_url = OSM2_API_BASE_URL + '{map}/section/{way_id}/{from_node_id}/{to_node_id}'
        way_info = requests.get(
            way_url.format(map=self.__latest_map, way_id=way_id, from_node_id=from_node_id, to_node_id=to_node_id))
        return way_info.json()
