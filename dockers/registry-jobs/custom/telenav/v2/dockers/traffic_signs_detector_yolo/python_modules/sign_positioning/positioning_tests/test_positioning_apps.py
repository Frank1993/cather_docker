from py_orbb_metadata import orbb_definitions_pb2, orbb_localization_pb2
from positioning_apps_runner import PositioningRunner

class Cluster(object):

    def __init__(self, way_id, type, signs_number, from_node_id, to_node_id):
        self.way_id = way_id
        self.type = type
        self.signs_number = signs_number
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id

    def __str__(self):
        out_string = "way id:" + str(self.way_id) + " type: " + self.type + " signs number: " + str(self.signs_number) \
                   + " from node: " + str(self.from_node_id) + " to node: " + str(self.to_node_id)
        return out_string

    def __eq__(self, other):
        return self.way_id == other.way_id and self.type == other.type and self.from_node_id == other.from_node_id and self.to_node_id == other.to_node_id

expectations = [
            Cluster(756524368,"SL_US_25",2, 431488039, 158663825)
           ,Cluster(19552422, "SL_US_25", 5, 17181782, 129530265)
           ,Cluster(19552421, "SL_US_25", 6, 129530265, 623161932)
           ,Cluster(19552420, "SL_US_25", 4, 903075814, 623161932)
           ,Cluster(19552255, "TURN_RESTRICTION_US_LEFT", 7, 737973058, 934708293)
           ,Cluster(108261578, "TURN_RESTRICTION_US_LEFT", 3, 664734729, 671928920)
           ,Cluster(732714284, "TURN_RESTRICTION_US_LEFT", 5, 707558888, 431488039)
           ,Cluster(19552421, "TURN_RESTRICTION_US_LEFT", 2, 623161932, 129530265)
           ,Cluster(19552421, "TURN_RESTRICTION_US_LEFT", 3, 129530265, 623161932)
           ]

def read_cluster(cluster_file_path):
    metadata = orbb_localization_pb2.Clusters()
    with open(cluster_file_path, 'rb') as f:
        metadata.ParseFromString(f.read())
    return metadata.cluster

def run_tests():
    positioning_runner = PositioningRunner()
    cluster_file_path = positioning_runner.run()
    clusters = read_cluster(cluster_file_path)
    number_of_matches = 0
    for expected_cluster in expectations:
        cluster_matched = False
        for received_cluster in clusters:
            actual_cluster = Cluster(received_cluster.way_id,
                                     orbb_definitions_pb2._MARK.values_by_number[received_cluster.type].name,
                                     received_cluster.nb_points, received_cluster.from_node_id, received_cluster.to_node_id)
            if expected_cluster == actual_cluster:
                cluster_matched = True
                continue
        if True == cluster_matched:
            number_of_matches = number_of_matches+1
        else:
            print '{}'.format('Cluster not matched: ')
            print str(expected_cluster)
    print '{} {} {} {}'.format('Test results pass rate ', str( number_of_matches), '/',  str(len(expectations)) )

def main():
    run_tests()

if __name__ == "__main__":
    main()