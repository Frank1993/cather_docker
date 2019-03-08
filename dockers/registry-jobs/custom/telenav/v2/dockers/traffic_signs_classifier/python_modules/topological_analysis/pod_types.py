from collections import namedtuple


GPSPoint = namedtuple('GPSPoint', ['latitude', 'longitude'])
TripInfo = namedtuple('TripInfo', ['image_index', 'timestamp', 'latitude', 'longitude', 'bearing'])
MatchInfo = namedtuple('MatchInfo', ['timestamp', 'way_id', 'from_node_id', 'to_node_id'])
MatchTuple = namedtuple('MatchTuple', ['original_point', 'match_point'])
MatchResult = namedtuple('MatchResult', ['match_tuple', 'signs'])