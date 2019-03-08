import requests
import arrow
import utils
import folium

from pod_types import *
from map_provider import MapProvider
from constants import *
from way import Way
from node import Node


class Matcher:
    OSV_DETAILS_URL = 'http://openstreetview.com/details'

    def __init__(self):
        self.__map_provider = MapProvider()
        self.__map_version = self.__map_provider.get_latest_map()
        self.match_url = OSM2_API_BASE_URL + '{map_version}/match'
        self.way_section_url = OSM2_API_BASE_URL + '{map_version}/section/{way_id}/{from_node_id}/{to_node_id}'
        self.snap_url = OSM2_API_BASE_URL + 'snap/{point_lat}/{point_lon}'

    def __get_trip_info(self, sequence):
        osv_details = requests.post(self.OSV_DETAILS_URL, data={'id': sequence}).json()
        trip_info = {}

        for first_picture, second_picture in zip(osv_details[OSV_KEY][PHOTOS_KEY],
                                                 osv_details[OSV_KEY][PHOTOS_KEY][1:]):
            timestamp = arrow.get(first_picture['shot_date']).timestamp
            trip_info[timestamp] = TripInfo(int(first_picture['sequence_index']),
                                            timestamp,
                                            float(first_picture[LAT_KEY]),
                                            float(first_picture[LNG_KEY]),
                                            utils.compute_bearing(
                                                GPSPoint(float(first_picture[LAT_KEY]),
                                                         float(first_picture[LNG_KEY])),
                                                GPSPoint(float(second_picture[LAT_KEY]),
                                                         float(second_picture[LNG_KEY])))
                                            )

        return trip_info

    def __convert_to_osm_squared_format(self, trip_info):
        trip_probes = {"probes": []}

        for info in trip_info.values():
            trip_probes["probes"].append({
                "timestamp": info.timestamp,
                "latitude": info.latitude,
                "longitude": info.longitude,
                "properties":
                    {"accuracy": 1,
                     "heading": info.bearing,
                     "speedMPH": 0}
            })

        return trip_probes

    def match_trip(self, trip_id):
        trip_info = self.__get_trip_info(trip_id)
        match_result = requests.post(self.match_url.format(map_version=self.__map_version),
                                     json=self.__convert_to_osm_squared_format(trip_info)).json()

        match_info = dict()
        for match_key, match_value in match_result[TIME_BASED_MATCHED_KEY].items():
            match_info[int(match_key)] = MatchInfo(int(match_key),
                                                   match_value[ID_KEY][WAY_ID],
                                                   match_value[ID_KEY][FROM_NODE_ID],
                                                   match_value[ID_KEY][TO_NODE_ID])
        return self.__create_matches(trip_info, match_info), match_info

    def __create_matches(self, original_trip, matched_trip):
        matches = dict()

        for original_instance_key, original_instance in original_trip.items():
            if original_instance_key in matched_trip:
                matches[original_instance.image_index] = self.__snap_point_to_way(
                    original_instance.latitude,
                    original_instance.longitude,
                    matched_trip[original_instance_key])
        return matches

    def __snap_point_to_way(self, point_lat, point_lon, match_info):
        way_geometry = self.__map_provider.get_way_info(match_info.way_id, match_info.from_node_id,
                                                        match_info.to_node_id)
        snap_pt = requests.post(self.snap_url.format(point_lat=point_lat, point_lon=point_lon),
                                json=way_geometry[GEOMETRY_KEY]).json()

        return MatchTuple(GPSPoint(point_lat, point_lon),
                          GPSPoint(snap_pt['location']['latitude'], snap_pt['location']['longitude']))


# this is just a dummy display to test folium... needs big improvements TODO
class MatchViewer:
    HTML_TRAFFIC_SIGN = '<img src="icons/traffic_lights.png" width="100" height="100">'
    HTML_GENERIC_SIGN = '<img src="icons/generic.jpg" width="100" height="100">'

    def __init__(self, output_path):
        self.__output_file = output_path + '/clanson.html'
        self.__map_provider = MapProvider()
        self.__map = folium.Map(max_zoom=30)

    def display_matched_sections(self, matched_ways, matched_points, signs):
        ways = list()

        for matched_way in matched_ways.values():
            way_id = matched_way.way_id
            from_node_id = matched_way.from_node_id
            to_node_id = matched_way.to_node_id

            way = Way(way_id)
            way_info = self.__map_provider.get_way_info(way_id, from_node_id, to_node_id)
            geometry = [GPSPoint(geom[LATITUDE_KEY], geom[LONGITUDE_KEY]) for geom in way_info[GEOMETRY_KEY]]
            way.add_geometry(geometry)

            ways.append(way)

        self.__display_ways(ways)
        self.__display_matched_points(matched_points, signs)

        self.__map.save(self.__output_file)

    def __display_ways(self, ways):
        for way in ways:
            coordinates = []
            for geom in way.geometry:
                coordinates.append([geom.latitude, geom.longitude])
                polyline = folium.PolyLine(locations=coordinates, weight=5)
                self.__map.add_child(polyline)

    def __display_matched_points(self, matched_points, signs):
        for matched_index, matched_instance in matched_points.items():
            if matched_index in signs:
                if signs[matched_index] == 'TRAFFIC_LIGHTS_SIGN':
                    html = folium.Html(self.HTML_TRAFFIC_SIGN, script=True)
                else:
                    html = folium.Html(self.HTML_GENERIC_SIGN, script=True)

                popup = folium.Popup(html, max_width=300)
                matched_pt = folium.Circle(
                    [matched_instance.match_point.latitude, matched_instance.match_point.longitude], radius=2,
                    popup=popup,
                    color='yellow', fill=True, fill_color='yellow')
            else:
                matched_pt = folium.Circle(
                    [matched_instance.match_point.latitude, matched_instance.match_point.longitude], radius=2,
                    color='green',
                    fill=True,
                    fill_color='green')

            original_pt = folium.Circle(
                [matched_instance.original_point.latitude, matched_instance.original_point.longitude], radius=2,
                color='red', fill=True, fill_color='red')

            polyline = folium.PolyLine(
                locations=[[matched_instance.original_point.latitude, matched_instance.original_point.longitude],
                           [matched_instance.match_point.latitude, matched_instance.match_point.longitude]],
                weight=4, color='yellow')

            self.__map.add_child(polyline)
            self.__map.add_child(original_pt)
            self.__map.add_child(matched_pt)
