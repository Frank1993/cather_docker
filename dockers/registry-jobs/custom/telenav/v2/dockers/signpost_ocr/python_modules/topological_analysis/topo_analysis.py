from detections_provider import DetectionsProvider
from matcher import Matcher, MatchViewer
from pod_types import GPSPoint
from map import OSMMap
import argparse

USERNAME = 'apollou'
PASSWORD = 'apollo123'
DB_URL = '10.230.7.115:3306'
DB_NAME = 'apollo_api_production'

parser = argparse.ArgumentParser()
parser.add_argument('--trip_id', type=int, help='trip id')
parser.add_argument('--output_path', type=str, help='output path')
args = parser.parse_args()

detroit_latitude = 42.331429
detroit_longitude = -83.045753
map = OSMMap(GPSPoint(detroit_latitude, detroit_longitude))

det = DetectionsProvider(USERNAME, PASSWORD, DB_URL, DB_NAME)
signs = det.get_all_detections_for_trip(args.trip_id)

print('match')
m = Matcher()
match_info, matched_ways = m.match_trip(args.trip_id)
print('view')

view = MatchViewer(args.output_path)
view.display_matched_sections(matched_ways, match_info, signs)



