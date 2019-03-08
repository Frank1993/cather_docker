import argparse
from functools import partial
from itertools import product
from multiprocessing import Pool
import pandas as pd
import tqdm

from apollo_python_common.lightweight_types import AttributeDict
import apollo_python_common.io_utils as io_utils
from sign_clustering import clustering
from sign_clustering.constants import *
from sign_clustering.evaluate_clusters import compare_clusters_to_gt


def get_score_for_params(params, detections_df):
    current_config = AttributeDict(params)
    filtered_df = clustering.filter_detections_by_distance(detections_df, current_config.roi_distance_threshold)
    filtered_df = clustering.filter_detections_by_gps_acc(filtered_df, current_config.gps_accuracy_threshold)
    _, clustered_df = clustering.get_clusters(filtered_df, current_config, threads_number=1)
    completeness_score, precision, recall, accuracy, distance_avg = compare_clusters_to_gt(clustered_df)
    return completeness_score


def get_best_params(detections_df, config, nr_threads):
    epsilon_list = range(15, 60)
    facing_factor_list = range(10, 45, 5)
    distance_threshold_list = range(50, 110, 10)
    #TODO: add list of values for gps accuracy
    params_list = list(product(epsilon_list, facing_factor_list, distance_threshold_list))
    config_list = []
    for params in params_list:
        new_config = dict()
        new_config.update(config)
        new_config['dbscan_epsilon'] = params[0]
        new_config['facing_factor'] = params[1]
        new_config['roi_distance_threshold'] = params[2]
        config_list.append(new_config)

    pool = Pool(nr_threads)
    score_list = list(tqdm.tqdm(pool.imap(partial(get_score_for_params,
                                                  detections_df=detections_df),
                                          config_list),
                                total=len(config_list)))
    pool.close()

    best_score_index = score_list.index(max(score_list))

    return score_list[best_score_index], params_list[best_score_index]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, required=True)
    parser.add_argument("-c", "--config_file", type=str, required=True)
    parser.add_argument("-g", "--ground_truth_file", type=str, required=True)
    parser.add_argument("-t", "--threads_nr", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    input_file = args.input_file
    config_file = args.config_file
    ground_truth_file = args.ground_truth_file
    nr_threads = args.threads_nr

    config = io_utils.config_load(config_file)

    rois_df = clustering.read_detections_input(input_file, config)
    gt_df = pd.read_csv(ground_truth_file)

    merged_df = pd.merge(gt_df, rois_df)

    best_score, best_params = get_best_params(merged_df, config, nr_threads)
    print("Best score: {}".format(best_score))
    print("Best params: {}".format(best_params))
