import argparse
import logging
import os

from tqdm import tqdm

tqdm.pandas()
import numpy as np

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util

import apollo_python_common.proto_api as proto_api

from object_detection.retinanet.generate_best_thresholds import BestThresholdEvaluator

from classification.scripts.roi_classifier.end_2_end_params import End_2_End_Params as e2e
from classification.scripts.constants import NetworkCfgParams as ncp
from classification.scripts.roi_classifier.roi_predictions_postprocessor import RoiPredictionsPostprocessor
from classification.scripts.roi_classifier.roi_dataset_creator import ROIDatasetCreator
from classification.run.dataset_creator import DatasetCreator
from classification.run.network_trainer import NetworkTrainer
import classification.scripts.network as network
import classification.scripts.validator as validator

import apollo_python_common.obj_detection_evaluator.protobuf_evaluator as protobuf_evaluator
from apollo_python_common.obj_detection_evaluator.model_statistics import ModelStatistics


class RoiClassifPipeline:
    def __init__(self, e2e_conf):
        self.e2e_conf = e2e_conf

    def __get_roi_dataset_creator_base_config(self, gt_folder, pred_file, output_folder_name, delta_perc_arr):
        config = {e2e.GT_FOLDER_KEY: gt_folder, e2e.PRED_ROIS_FILE_KEY: pred_file,
                  e2e.BASE_OUTPUT_PATH_KEY: os.path.join(self.e2e_conf[e2e.OUTPUT_PATH_KEY],
                                                         e2e.DATASET_FOLDER_NAME,
                                                         output_folder_name), e2e.DELTA_PERC_ARR_KEY: delta_perc_arr,
                  e2e.MIN_SIZE_KEY: self.e2e_conf[e2e.MIN_SIZE_KEY],
                  e2e.IOU_THRESHOLD_KEY: self.e2e_conf[e2e.IOU_THRESHOLD_KEY],
                  e2e.DATASET_CLASSES_FILE_KEY: self.e2e_conf[e2e.DATASET_CLASSES_FILE_KEY]}
        return config

    def __get_roi_dataset_train_config(self):
        return self.__get_roi_dataset_creator_base_config(gt_folder=self.e2e_conf[e2e.TRAIN_GT_FOLDER_KEY],
                                                          pred_file=self.e2e_conf[e2e.TRAIN_PREDS_FILE_KEY],
                                                          output_folder_name=e2e.TRAIN_DATASET_FOLDER_NAME,
                                                          delta_perc_arr=[-0.25, 0, 0.25])

    def __get_roi_dataset_train_temp_config(self):
        return self.__get_roi_dataset_creator_base_config(gt_folder=self.e2e_conf[e2e.TRAIN_GT_FOLDER_KEY],
                                                          pred_file=self.e2e_conf[e2e.TRAIN_PREDS_FILE_KEY],
                                                          output_folder_name=e2e.TRAIN_TEMP_DATASET_FOLDER_NAME,
                                                          delta_perc_arr=[0])

    def __get_roi_dataset_test_config(self):
        return self.__get_roi_dataset_creator_base_config(gt_folder=self.e2e_conf[e2e.TEST_GT_FOLDER_KEY],
                                                          pred_file=self.e2e_conf[e2e.TEST_PREDS_FILE_KEY],
                                                          output_folder_name=e2e.TEST_DATASET_FOLDER_NAME,
                                                          delta_perc_arr=[0])

    def __generate_roi_dataset_images(self):
        print("Generating cropped images from rois...")
        train_config = self.__get_roi_dataset_train_config()
        train_temp_config = self.__get_roi_dataset_train_temp_config()
        test_config = self.__get_roi_dataset_test_config()
        all_configs = [train_config, train_temp_config, test_config]

        for config in all_configs:
            print(config[e2e.BASE_OUTPUT_PATH_KEY])
            io_utils.create_folder(config[e2e.BASE_OUTPUT_PATH_KEY])
            ROIDatasetCreator(config).create_dataset()

    def __get_dataset_builder_config(self, base_path, with_augment=False):
        builder_config = {}
        builder_config[e2e.BASE_PATH_KEY] = base_path
        builder_config[e2e.IMG_DIM_KEY] = self.e2e_conf[e2e.CLASSIF_IMG_DIM_KEY]
        builder_config[e2e.NR_IMAGES_PER_CLASS_KEY] = 999999
        builder_config[e2e.AUGMENT_STRATEGIES_KEY] = {
            e2e.FLIP_KEY: with_augment,
            e2e.ROTATE_CW_KEY: with_augment,
            e2e.ROTATE_CCW_KEY: with_augment
        }
        builder_config[e2e.SUFFIX_KEY] = "AUTO"
        builder_config[e2e.WITH_VP_CROP_KEY] = False
        builder_config[e2e.SPLIT_TRAIN_TEST_BY_SEQ_ID_KEY] = False
        builder_config[e2e.SQUARE_IMAGE_KEY] = True
        builder_config[e2e.KEEP_ASPECT_KEY] = False

        return builder_config

    def __get_train_dataset_builder_config(self):
        base_path = os.path.join(self.e2e_conf[e2e.OUTPUT_PATH_KEY],
                                 e2e.DATASET_FOLDER_NAME,
                                 e2e.TRAIN_DATASET_FOLDER_NAME)

        return self.__get_dataset_builder_config(base_path, with_augment=True)

    def __get_train_temp_dataset_builder_config(self):
        base_path = os.path.join(self.e2e_conf[e2e.OUTPUT_PATH_KEY],
                                 e2e.DATASET_FOLDER_NAME,
                                 e2e.TRAIN_TEMP_DATASET_FOLDER_NAME)
        return self.__get_dataset_builder_config(base_path)

    def __get_test_dataset_builder_config(self):
        base_path = os.path.join(self.e2e_conf[e2e.OUTPUT_PATH_KEY],
                                 e2e.DATASET_FOLDER_NAME,
                                 e2e.TEST_DATASET_FOLDER_NAME)
        return self.__get_dataset_builder_config(base_path)

    def __build_datasets(self):
        print("Building datasets for classifier....")
        train_config = self.__get_train_dataset_builder_config()
        train_temp_config = self.__get_train_temp_dataset_builder_config()
        test_config = self.__get_test_dataset_builder_config()

        all_configs = [train_config, train_temp_config, test_config]

        for config in all_configs:
            print(config[e2e.BASE_PATH_KEY])
            DatasetCreator(config).create_dataset()

    def __get_generated_dataset_folder_name(self, config):
        return "classes=2_imgs={}_size={}_{}".format(config[e2e.NR_IMAGES_PER_CLASS_KEY],
                                                     config[e2e.IMG_DIM_KEY],
                                                     config[e2e.SUFFIX_KEY])

    def __get_full_dataset_path(self, config, dataset_folder_name):
        generated_dataset_folder = self.__get_generated_dataset_folder_name(config)
        return os.path.join(self.e2e_conf[e2e.OUTPUT_PATH_KEY],
                            e2e.DATASET_FOLDER_NAME,
                            dataset_folder_name,
                            e2e.PROCESSED_FOLDER_NAME,
                            generated_dataset_folder,
                            e2e.PARAMS_FOLDER_NAME,
                            e2e.MODEL_PARAMS_FILE_NAME)

    def __get_network_train_config(self):
        network_config = {}

        network_config[ncp.WEIGHTS_LOCAL_OUTPUT_PATH_PARAM] = os.path.join(self.e2e_conf[e2e.OUTPUT_PATH_KEY],
                                                                           e2e.NETWORK_FOLDER_NAME,
                                                                           e2e.WEIGHTS_FOLDER_NAME)
        network_config[ncp.FTP_OUTPUT_PARAM] = self.e2e_conf[e2e.FTP_OUTPUT_PATH_KEY]
        network_config[ncp.NR_EPOCHS_PARAM] = self.e2e_conf[e2e.NR_EPOCHS_KEY]
        network_config[ncp.WITH_DIFFERENT_TEST_SET_PARAM] = True

        network_config[ncp.NETWORK_CONFIG_PARAM] = {
            ncp.NR_CONV_BLOCKS_PARAM: 3,
            ncp.NR_FILTERS_PARAM: 32,
            ncp.DROPOUT_LEVEL_PARAM: 0.6
        }

        network_config[ncp.TRAIN_MODEL_PARAMS_PATH] = self.__get_full_dataset_path(
            self.__get_train_dataset_builder_config(),
            e2e.TRAIN_DATASET_FOLDER_NAME)
        network_config[ncp.TEST_MODEL_PARAMS_PATH] = self.__get_full_dataset_path(
            self.__get_test_dataset_builder_config(),
            e2e.TEST_DATASET_FOLDER_NAME)
        return network_config

    def __train_network(self):
        print("Training the classifier...")
        network_config = self.__get_network_train_config()
        NetworkTrainer(network_config).train_network()

    def __predict_on_data(self, model, df_path_list, img_path_list, conv_path_list, output_pred_name):
        pred_data_df = network.make_prediction_on_dataset(df_path_list, img_path_list, conv_path_list,
                                                          model,
                                                          nr_batches=None,
                                                          with_img=False)

        print("Image accuracy = %f" % (validator.compute_accuracy(pred_data_df)))

        pred_output_path = os.path.join(self.e2e_conf[e2e.OUTPUT_PATH_KEY],
                                        e2e.PREDICTIONS_FOLDER_NAME)
        io_utils.create_folder(pred_output_path)
        pred_data_df.to_pickle(os.path.join(pred_output_path,
                                            output_pred_name))

    def __predict_on_train_test(self):
        print("Predicting on train/test set...")
        network_config = self.__get_network_train_config()
        network_config[ncp.TRAIN_MODEL_PARAMS_PATH] = \
            self.__get_full_dataset_path(self.__get_train_temp_dataset_builder_config(),
                                         e2e.TRAIN_TEMP_DATASET_FOLDER_NAME)

        network_trainer = NetworkTrainer(network_config)

        model = network_trainer.build_model()
        model.load_weights(os.path.join(network_config[ncp.WEIGHTS_LOCAL_OUTPUT_PATH_PARAM],
                                        e2e.WEIGHTS_FILE_NAME))

        train_df_path_list, train_img_path_list, train_conv_path_list, \
        test_df_path_list, test_img_path_list, test_conv_path_list = network_trainer.get_data_paths()

        self.__predict_on_data(model, train_df_path_list, train_img_path_list, train_conv_path_list,
                               e2e.TRAIN_PREDS_FILE_NAME)
        self.__predict_on_data(model, test_df_path_list, test_img_path_list, test_conv_path_list,
                               e2e.TEST_PREDS_FILE_NAME)

    def __postprocessed_rois(self, initial_roi_path, classif_pred_name, post_proc_name, conf_thresholds=None):
        classif_pred_path = os.path.join(self.e2e_conf[e2e.OUTPUT_PATH_KEY],
                                         e2e.PREDICTIONS_FOLDER_NAME,
                                         classif_pred_name)

        output_folder = os.path.join(self.e2e_conf[e2e.OUTPUT_PATH_KEY],
                                     e2e.POSTPROCESS_FOLDER_NAME,
                                     post_proc_name)

        RoiPredictionsPostprocessor(initial_roi_path, classif_pred_path, output_folder) \
            .postprocess_predictions(conf_thresholds)

    def __generate_postprocessed_rois(self):
        print("Generating the postprocessed rois...")
        self.__postprocessed_rois(self.e2e_conf[e2e.TRAIN_PREDS_FILE_KEY],
                                  e2e.TRAIN_PREDS_FILE_NAME,
                                  e2e.TRAIN_FOLDER_NAME,
                                  [0.5])

        self.__postprocessed_rois(self.e2e_conf[e2e.TEST_PREDS_FILE_KEY],
                                  e2e.TEST_PREDS_FILE_NAME,
                                  e2e.TEST_FOLDER_NAME)

    def __compute_best_thresholds(self, gt_rois_file, pred_rois_file, min_size, result_file):
        io_utils.create_folder(os.path.dirname(result_file))
        gt_dict = proto_api.create_images_dictionary(proto_api.read_imageset_file(gt_rois_file))
        pred_dict = proto_api.create_images_dictionary(proto_api.read_imageset_file(pred_rois_file))

        thresh_eval = BestThresholdEvaluator(gt_dict, pred_dict, min_size=min_size)
        thresholds_per_class = thresh_eval.get_best_thresholds()
        thresh_eval.evaluate_with_best_thresholds(thresholds_per_class)

        io_utils.json_dump(thresholds_per_class, result_file)

    def __generate_best_thresholds(self):
        print("Generating best thresholds...")
        self.__compute_best_thresholds(
            gt_rois_file=os.path.join(self.e2e_conf[e2e.TRAIN_GT_FOLDER_KEY], e2e.ROIS_FILE_NAME),
            pred_rois_file=os.path.join(self.e2e_conf[e2e.OUTPUT_PATH_KEY],
                                        e2e.POSTPROCESS_FOLDER_NAME,
                                        e2e.TRAIN_FOLDER_NAME,
                                        "postprocessed_preds_0.5",
                                        e2e.ROIS_FILE_NAME),
            min_size=self.e2e_conf[e2e.MIN_SIZE_KEY],
            result_file=os.path.join(self.e2e_conf[e2e.OUTPUT_PATH_KEY],
                                     e2e.BEST_THRESHOLDS_FOLDER_NAME,
                                     e2e.BEST_THRESHOLDS_FILE_NAME))

    def __print_stats(self, model_stats):
        for key, statistics_element in list(model_stats.statistics.items())[-1:]:
            precision = statistics_element.precision()
            recall = statistics_element.recall()
            accuracy = statistics_element.accuracy()

            print("\t Precision {:.3f}".format(precision))
            print("\t Recall    {:.3f}".format(recall))
            print("\t Accuracy  {:.3f}".format(accuracy))

    def __evaluate(self,
                   expected_rois_file,
                   actual_rois_file,
                   selected_classes_file,
                   classes_thresholds_file,
                   min_size=25):

        expected_detection_dictionary, actual_detection_dictionary = protobuf_evaluator.get_data_dicts(
            expected_rois_file,
            actual_rois_file,
            selected_classes_file,
            classes_thresholds_file)

        model_statistics = ModelStatistics(expected_detection_dictionary, actual_detection_dictionary, min_size)
        model_statistics.compute_model_statistics()
        return model_statistics

    def __output_results(self, best_conf, best_model_stats):
        statistics_path = os.path.join(self.e2e_conf[e2e.OUTPUT_PATH_KEY],
                                       e2e.RESULTS_FOLDER_NAME,
                                       e2e.STATISTICS_FILE_NAME)

        io_utils.create_folder(os.path.dirname(statistics_path))
        best_model_stats.output_statistics(statistics_path)

        best_conf_path = os.path.join(self.e2e_conf[e2e.OUTPUT_PATH_KEY],
                                      e2e.POSTPROCESS_FOLDER_NAME,
                                      e2e.TEST_FOLDER_NAME,
                                      "postprocessed_preds_{}".format(best_conf),
                                      e2e.ROIS_FILE_NAME)

        best_results_path = os.path.join(self.e2e_conf[e2e.OUTPUT_PATH_KEY],
                                         e2e.RESULTS_FOLDER_NAME,
                                         e2e.BEST_PATH_FILE_NAME)

        with open(best_results_path, 'w') as f:
            f.write(best_conf_path + '\n')

    def __get_accuracy(self, model_stats):
        return list(model_stats.statistics.items())[-1][1].accuracy()

    def __perform_error_analysis(self):
        print("Performing error analysis...")
        best_conf = 0
        best_model_stats = None

        pred_path = os.path.join(self.e2e_conf[e2e.OUTPUT_PATH_KEY],
                                 e2e.POSTPROCESS_FOLDER_NAME,
                                 e2e.TEST_FOLDER_NAME,
                                 "postprocessed_preds_{}",
                                 e2e.ROIS_FILE_NAME)

        conf_thresholds = [round(x, 3) for x in np.arange(0.3, 1.0, 0.05)]

        for conf_threshold in tqdm(conf_thresholds):
            model_stats = self.__evaluate(
                expected_rois_file=os.path.join(self.e2e_conf[e2e.TEST_GT_FOLDER_KEY],
                                                e2e.ROIS_FILE_NAME),
                min_size=self.e2e_conf[e2e.MIN_SIZE_KEY],
                actual_rois_file=pred_path.format(conf_threshold),
                selected_classes_file=self.e2e_conf[e2e.EVALUATE_CLASSES_FILE_KEY],
                classes_thresholds_file=os.path.join(self.e2e_conf[e2e.OUTPUT_PATH_KEY],
                                                     e2e.BEST_THRESHOLDS_FOLDER_NAME,
                                                     e2e.BEST_THRESHOLDS_FILE_NAME))

            print(conf_threshold)
            self.__print_stats(model_stats)

            if best_model_stats is None or self.__get_accuracy(model_stats) > self.__get_accuracy(best_model_stats):
                best_conf = conf_threshold
                best_model_stats = model_stats

        print("-------------BEST SCORES----------")
        print(best_conf)

        self.__print_stats(best_model_stats)
        self.__output_results(best_conf, best_model_stats)

    def run_pipeline(self):
        self.__generate_roi_dataset_images()
        self.__build_datasets()
        self.__train_network()
        self.__predict_on_train_test()
        self.__generate_postprocessed_rois()
        self.__generate_best_thresholds()
        self.__perform_error_analysis()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config_file", help="path to config json",
                        type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':

    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    end_2_end_conf = io_utils.json_load(args.config_file)

    try:
        RoiClassifPipeline(end_2_end_conf).run_pipeline()
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
