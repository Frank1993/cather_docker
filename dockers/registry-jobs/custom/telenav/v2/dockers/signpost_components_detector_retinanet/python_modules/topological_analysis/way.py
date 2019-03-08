from enum import Enum
from utils import *
from node import GPSPoint
from constants import *
from node import Node

class Highway(Enum):
    MOTORWAY = 0
    TRUNK = 1
    PRIMARY = 2
    SECONDARY = 3
    TERTIARY = 4
    RESIDENTIAL = 5
    MOTORWAY_LINK = 6
    TRUNK_LINK = 7
    PRIMARY_LINK = 8
    SECONDARY_LINK = 9
    TERTIARY_LINK = 10
    NONE = 11


class Way:
    def __init__(self, osm_id, from_node=Node(), to_node=Node()):
        self.osm_id = osm_id
        self.from_node = from_node
        self.to_node = to_node
        self.heading = compute_bearing(from_node.gps_point, to_node.gps_point)
        self.geometry = list()

    def add_geometry(self, geometry):
        self.geometry = geometry

    def __str__(self):
        return "way: " + str(self.osm_id) + " - (" + str(self.from_node.osm_id) + ", " + str(self.to_node.osm_id) + ")"

    def __eq__(self, other):
        return self.osm_id == other.osm_id and self.from_node.osm_id == other.from_node.osm_id and self.to_node.osm_id == other.to_node.osm_id

    def __hash__(self):
        return hash((self.osm_id, self.from_node.osm_id, self.to_node.osm_id))


highway_map = {
    'motorway': Highway.MOTORWAY,
    'trunk': Highway.TRUNK,
    'primary': Highway.PRIMARY,
    'secondary': Highway.SECONDARY,
    'tertiary': Highway.TERTIARY,
    'residential': Highway.RESIDENTIAL,
    'motorway_link': Highway.MOTORWAY_LINK,
    'trunk_link': Highway.TRUNK_LINK,
    'primary_link': Highway.PRIMARY_LINK,
    'secondary_link': Highway.SECONDARY_LINK,
    'tertiary_link': Highway.TERTIARY_LINK,
    'none': Highway.NONE
}


def transform_ways(way_json):
    way_identifier = way_json['identifier']
    way_geometry = way_json['geometry']
    return Way(way_identifier[WAY_ID],
               Node(way_identifier[FROM_NODE_ID], GPSPoint(way_geometry[0][LATITUDE_KEY], way_geometry[0][LONGITUDE_KEY])),
               Node(way_identifier[TO_NODE_ID], GPSPoint(way_geometry[-1][LATITUDE_KEY], way_geometry[-1][LONGITUDE_KEY])))
