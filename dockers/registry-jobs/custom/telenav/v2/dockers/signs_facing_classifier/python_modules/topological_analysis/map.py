from way import *
from node import *
from map_provider import *
from utils import *


class OSMMap:
    LANE_ANGLE_EPSILON = 10

    def __init__(self, center_point):
        map_provider = MapProvider()
        ways_json = map_provider.get_near_ways(center_point)

        self.__ways = self.__get_ways(ways_json)
        self.__nodes = self.__create_nodes_list()

    def __get_ways(self, ways_json):
        return list(map(transform_ways, ways_json))

    def __create_nodes_list(self):
        nodes = dict()
        for way in self.__ways:
            if way.from_node.osm_id not in nodes:
                nodes[way.from_node.osm_id] = way.from_node

            if way.to_node.osm_id not in nodes:
                nodes[way.to_node.osm_id] = way.to_node

            nodes[way.from_node.osm_id].add_out_way(way)
            nodes[way.to_node.osm_id].add_in_way(way)

        return nodes

    def __show_ways(self):
        for way in self.__ways:
            node = self.__nodes[way.to_node.osm_id]
            print('for way: ' + str(way))
            for to_way in node.out_ways:
                if way == to_way:
                    continue
                if self.__is_way_left(way.heading, to_way.heading):
                    print('    way right: ' + str(to_way))
                elif self.__is_way_right(way.heading, to_way.heading):
                    print('    way right: ' + str(to_way))
                elif self.__is_way_left(way.heading, to_way.heading):
                    print('    way straight: ' + str(to_way))
                else:
                    print('    way uturn: ' + str(to_way))

            for to_way in node.in_ways:
                if way == to_way:
                    continue
                if self.__is_way_left(way.heading, revert_heading(to_way.heading)):
                    print('    way left: ' + str(to_way))
                elif self.__is_way_right(way.heading, revert_heading(to_way.heading)):
                    print('    way right: ' + str(to_way))
                elif self.__is_way_left(way.heading, revert_heading(to_way.heading)):
                    print('    way straight: ' + str(to_way))
                else:
                    print('    way uturn: ' + str(to_way))

    def __is_way_right(self, from_way_heading, to_way_heading):
        delta_angle = from_way_heading - to_way_heading
        return ((delta_angle < 0 - self.LANE_ANGLE_EPSILON) and (delta_angle > -180 + self.LANE_ANGLE_EPSILON)) or (
            delta_angle > 180 + self.LANE_ANGLE_EPSILON)

    def __is_way_left(self, from_way_heading, to_way_heading):
        delta_angle = from_way_heading - to_way_heading
        return ((delta_angle > self.LANE_ANGLE_EPSILON) and (delta_angle < 180 - self.LANE_ANGLE_EPSILON)) or (
            delta_angle < -180 - self.LANE_ANGLE_EPSILON)

    def __is_straight_way(self, from_way_heading, to_way_heading):
        delta_angle = from_way_heading - to_way_heading
        return (delta_angle >= 0 - self.LANE_ANGLE_EPSILON) and (delta_angle <= self.LANE_ANGLE_EPSILON)
