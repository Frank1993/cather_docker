import argparse
import logging

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as proto_api
import apollo_python_common.ml_pipeline.config_api as config_api
import classification.scripts.utils as utils
from classification.scripts.prediction.abstract_classif_mq_predictor import AbstractClassifPredictor
from apollo_python_common.ml_pipeline.config_api import MQ_Param


class RoiClassifPredictor(AbstractClassifPredictor):
    BAD_CLASS_NAME = "bad"
    GOOD_CLASS_NAME = "good"
    EXCLUDEDED_ROIS_ES_KEY = "classifier_excluded_rois"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.conf_thresh = config_api.get_config_param(MQ_Param.ROI_BAD_CLASS_CONF_THRESHOLD, self.config,
                                                       default_value=0.5)

    def __resize_cropped_roi(self, full_img, roi_proto):

        roi_img, roi_id = utils.extract_cropped_roi(full_img, roi_proto)
        roi_img = self.resize(roi_img)
        roi_img = self.preprocess_image_according_to_backbone(roi_img)

        return roi_img, roi_id

    def __predict_batch(self, rois_2_ids):
        if len(rois_2_ids) == 0:
            return []

        rois, ids = zip(*rois_2_ids)
        predictions = self.predict_with_model(rois)

        predictions_2_ids = list(zip(predictions, ids))

        return predictions_2_ids

    def __is_bad_prediction(self, preds):
        class_name_2_confidence = {}
        for index, confidence in enumerate(preds):
            class_name = self.index_2_params[0].classIndex_2_class[index]
            class_name_2_confidence[class_name] = confidence

        return class_name_2_confidence[self.BAD_CLASS_NAME] > self.conf_thresh

    def __filter_bad_predictions(self, predictions_2_ids):
        return [(p, i) for p, i in predictions_2_ids if self.__is_bad_prediction(p)]

    def __remove_bad_rois_inplace(self, image_proto, bad_predictions_2_ids):

        bad_ids = [i for _, i in bad_predictions_2_ids]

        image_rois = image_proto.rois

        for roi_proto in image_rois:
            roi_id = utils.roi_2_id(roi_proto)
            if roi_id in bad_ids:
                image_rois.remove(roi_proto)

    def __get_class_name_with_confidence(self, pred, index):
        return "{}:{}".format(self.index_2_params[0].classIndex_2_class[index], round(float(pred[index]), 4))

    def __audit_bad_rois(self, image_proto, bad_predictions_2_ids):
        pred_2_ids_str = ["{},{},{}".format(roi_id,
                                            self.__get_class_name_with_confidence(pred, 0),
                                            self.__get_class_name_with_confidence(pred, 1))
                          for pred, roi_id in bad_predictions_2_ids]

        super().set_audit_key_val(image_proto.metadata.image_path, self.EXCLUDEDED_ROIS_ES_KEY, pred_2_ids_str)

    def preprocess(self, image_proto):

        rois_proto = [roi for roi in image_proto.rois if proto_api.get_roi_type_name(roi.type) != "SIGNPOST_GENERIC"]

        if len(rois_proto) == 0:
            return []

        img = self.read_image(image_proto)
        rois_2_ids = [self.__resize_cropped_roi(img, roi_proto) for roi_proto in rois_proto]

        return rois_2_ids

    def predict(self, rois_2_ids_lists):

        if self.model is None:
            self.model = self.load_model()

        predictions_2_ids_list = [self.__predict_batch(rois_2_ids) for rois_2_ids in rois_2_ids_lists]

        return predictions_2_ids_list

    def postprocess(self, predictions_2_ids, image_proto):

        bad_predictions_2_ids = self.__filter_bad_predictions(predictions_2_ids)
        self.__remove_bad_rois_inplace(image_proto, bad_predictions_2_ids)
        self.__audit_bad_rois(image_proto, bad_predictions_2_ids)
        return image_proto


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="config file path", type=str, required=True)

    return parser.parse_args()


def run_predictor(conf_file):
    conf = io_utils.config_load(conf_file)
    predictor = RoiClassifPredictor(conf)
    predictor.start()


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    args = parse_arguments()
    run_predictor(args.config_file)
