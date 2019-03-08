import argparse
import sys
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing

from apollo_python_common.obj_detection_evaluator.model_statistics import ModelStatistics
import apollo_python_common.obj_detection_evaluator.protobuf_evaluator as proto_eval
import apollo_python_common.io_utils as io_utils
import apollo_python_common.proto_api as proto_api


class BestThresholdEvaluator:

    def __init__(self, gt_dict, pred_dict, min_size):
        self.gt_dict = gt_dict
        self.pred_dict = pred_dict
        self.min_size = min_size

    def __compute_statistics(self, target_gt_dict, target_pred_dict):
        model_statistics = ModelStatistics(proto_eval.convert_rois_dictionary(target_gt_dict),
                                           proto_eval.convert_rois_dictionary(target_pred_dict),
                                           self.min_size)
        model_statistics.compute_model_statistics()

        return model_statistics.statistics

    def __filter_by_class(self, d, class_name):
        return proto_api.filter_rois_by_classes([class_name], d)

    def __filter_by_confidence(self, d, class_name, thr):
        return proto_api.get_confident_rois(d, {class_name: thr})

    def get_threshold_for_class(self, class_name):
        class_gt_dict = self.__filter_by_class(self.gt_dict, class_name)
        class_pred_dict = self.__filter_by_class(self.pred_dict, class_name)

        best_acc, best_threshold = 0, 0.1

        for thr in tqdm(np.arange(0.1, 1.01, 0.05)):
            confident_class_pred_dict = self.__filter_by_confidence(class_pred_dict, class_name, round(thr, 2))
            total_statistic = self.__compute_statistics(class_gt_dict, confident_class_pred_dict)['Total']
            accuracy = total_statistic.accuracy()
            has_preds = total_statistic.true_positives + total_statistic.false_positives + total_statistic.miss_classified != 0
            if has_preds and accuracy >= best_acc:
                best_acc = accuracy
                best_threshold = round(thr, 2)

        if best_threshold == 1:  # accuracy = 0 for all thresholds
            best_threshold = 0.1

        print('Best threshold for class {}: {}'.format(class_name, best_threshold))
        return class_name, best_threshold

    def __print_stats(self, statistics_dict):
        for key, statistics_element in statistics_dict.items():
            print("{} {}".format(key, statistics_element))

    def __get_all_classes(self):
        return set(proto_api.get_class_names_from_images_dictionary(self.gt_dict)) \
            .union(set(proto_api.get_class_names_from_images_dictionary(self.pred_dict)))

    def get_best_thresholds(self):

        threads_number = multiprocessing.cpu_count() // 2
        pool = Pool(threads_number)
        thresholds = pool.map(self.get_threshold_for_class, self.__get_all_classes())

        return dict(thresholds)

    def evaluate_with_best_thresholds(self, thresholds_per_class):
        confident_dict = proto_api.get_confident_rois(self.pred_dict, thresholds_per_class)
        statistics_dict = self.__compute_statistics(self.gt_dict, confident_dict)
        self.__print_stats(statistics_dict)

def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate classes confidence thresholds for best accuracy.')

    parser.add_argument("-g", "--gt_rois_file",
                        type=str, required=True)
    parser.add_argument("-p", "--pred_rois_file",
                        type=str, required=True)
    parser.add_argument("-o", "--result_file", type=str, required=True,
                        help='The file where to store the pickled dictionary with thresholds.',
                        default='./classes_thresholds.pkl')
    parser.add_argument("-c", "--selected_classes_file",
                        type=str, required=False, default=None)
    parser.add_argument("-m", "--min_size",
                        type=int, required=False, default=25)
    return parser.parse_args(args)


def main():
    args = sys.argv[1:]
    args = parse_args(args)
    gt_dict = proto_api.create_images_dictionary(proto_api.read_imageset_file(args.gt_rois_file))
    pred_dict = proto_api.create_images_dictionary(proto_api.read_imageset_file(args.pred_rois_file))

    if args.selected_classes_file:
        selected_classes = io_utils.json_load(args.selected_classes_file).selected_classes
        gt_dict = proto_api.filter_rois_by_classes(selected_classes, gt_dict)
        pred_dict = proto_api.filter_rois_by_classes(selected_classes, pred_dict)

    thresh_eval = BestThresholdEvaluator(gt_dict, pred_dict, min_size=args.min_size)
    thresholds_per_class = thresh_eval.get_best_thresholds()
    thresh_eval.evaluate_with_best_thresholds(thresholds_per_class)

    io_utils.json_dump(thresholds_per_class, args.result_file)


if __name__ == '__main__':
    main()
