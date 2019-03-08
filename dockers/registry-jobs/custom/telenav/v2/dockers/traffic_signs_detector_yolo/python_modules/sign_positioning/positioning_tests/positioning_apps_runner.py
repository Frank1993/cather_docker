import ConfigParser
from runners.first_local_runner import FirstLocalRunner
from runners.cluster_runner import ClusterRunner
from runners.geometry_matcher_runner import GeometryMatcherRunner

class PositioningRunner(object):

    def run(self):
        config = ConfigParser.RawConfigParser()
        config.read('../conf/test_positioning_apps.cfg')

        first_local_app_path = config.get("first_local_app_arguments", 'first_local_app_path')
        first_local_app_input = config.get("first_local_app_arguments", 'first_local_app_input')
        first_local_app_output = config.get("first_local_app_arguments", 'first_local_app_output')
        cluster_app_path = config.get("cluster_app_arguments", 'cluster_app_path')
        cluster_app_input = config.get("cluster_app_arguments", 'cluster_app_input')
        cluster_app_output = config.get("cluster_app_arguments", 'cluster_app_output')
        geometry_matcher_path = config.get("geometry_matcher_arguments", 'geometry_matcher_path')
        geometry_matcher_input_proto = config.get("geometry_matcher_arguments", 'geometry_matcher_input_proto')
        geometry_matcher_map_path = config.get("geometry_matcher_arguments", 'geometry_matcher_map_path')
        geometry_matcher_output = config.get("geometry_matcher_arguments", 'geometry_matcher_output')

        first_local_runner = FirstLocalRunner(first_local_app_path, first_local_app_input, first_local_app_output)
        first_local_runner()

        cluster_runner = ClusterRunner(cluster_app_path, cluster_app_input,"", cluster_app_output)
        cluster_runner()

        geometry_matcher_runner = GeometryMatcherRunner(geometry_matcher_path, geometry_matcher_input_proto,
                                                        geometry_matcher_map_path, geometry_matcher_output)
        geometry_matcher_runner()
        return geometry_matcher_output
