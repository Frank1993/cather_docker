from pod_types import GPSPoint


class Node:
    def __init__(self, osm_id=0, gps_point=GPSPoint(0, 0)):
        self.osm_id = osm_id
        self.gps_point = gps_point
        self.in_ways = list()
        self.out_ways = list()

    def add_way(self, way):
        if way.to_node == self:
            self.in_ways.append(way)
        else:
            self.out_ways.append(way)

    def add_in_way(self, way):
        self.in_ways.append(way)

    def add_out_way(self, way):
        self.out_ways.append(way)

    def __eq__(self, other):
        return self.osm_id == other.osm_id

    def __hash__(self):
        return self.osm_id
