import sys
import argparse
import logging

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
from apollo_python_common.ml_pipeline.multi_threaded_predictor import MultiThreadedPredictor
from apollo_python_common.proto_api import MQ_Messsage_Type
from sign_clustering import clustering



class SignClusteringPredictor(MultiThreadedPredictor):
    '''
    Multi threaded predictor for sign clustering
    '''
    EMPTY_IMAGE_SET_WARNING = "Empty image set in geotile proto"

    def __init__(self, config, **kwargs):
        super().__init__(config, mq_message_type=MQ_Messsage_Type.GEO_TILE, **kwargs)
        self.log_level = 0

    def preprocess(self, geotile_proto):
        image_set_df = clustering.image_set_to_dataframe(geotile_proto.image_set, config=self.config)
        image_set_df = clustering.filter_detections_by_distance(image_set_df, self.config.roi_distance_threshold)
        image_set_df = clustering.filter_detections_by_gps_acc(image_set_df, self.config.gps_accuracy_threshold)

        if image_set_df.shape[0] == 0:
            super().log_audit_warning(geotile_proto, self.EMPTY_IMAGE_SET_WARNING)

        return image_set_df

    def predict(self, input_msg_list):
        predictions_list = []
        for image_set_df in input_msg_list:
            if image_set_df.shape[0] == 0:
                predictions_list.append(None)
                continue
            _, clusters_df = clustering.get_clusters(image_set_df, self.config, threads_number=1)
            proto_clusters = clustering.create_cluster_proto(clusters_df, self.config)
            predictions_list.append(proto_clusters)
        return predictions_list

    def postprocess(self, one_geotile_clusters, geotile_proto):
        if one_geotile_clusters is not None:
            geotile_proto.clusters.extend(one_geotile_clusters.cluster)
        return geotile_proto


def run_predictor(conf_file):
    config = io_utils.config_load(conf_file)
    predictor = SignClusteringPredictor(config)
    predictor.start()


def __parse_args(args):
    parser = argparse.ArgumentParser(description='Sign clustering. Takes a geotile as input, processes the '
                                                 'image set contained and returns the same geotile with the '
                                                 'clusters computed')
    parser.add_argument('--config_file',
                        help='Configuration file''s path',
                        default='../config/clustering_config.json')
    return parser.parse_args(args)


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    # parse arguments
    args = sys.argv[1:]
    args = __parse_args(args)
    run_predictor(args.config_file)
