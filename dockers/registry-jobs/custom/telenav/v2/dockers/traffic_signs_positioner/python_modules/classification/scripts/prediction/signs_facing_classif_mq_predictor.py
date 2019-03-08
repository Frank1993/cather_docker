import argparse
import logging
import os

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as proto_api
import classification.scripts.signs_facing_classifier.postprocess as postprocess
import classification.scripts.utils as utils
from classification.scripts.prediction.abstract_classif_mq_predictor import AbstractClassifPredictor
from classification.scripts.signs_facing_classifier import constants as sf_constants
from classification.scripts.signs_facing_classifier.constants import SignFacingLabel


class SignsFacingClassifPredictor(AbstractClassifPredictor):
    """ Predictor for the signs facing classification component. """

    ALG_NAME = "signs_facing_classifier"
    ES_KEY_ROI_FACING = "roi_facing_prediction"
    BOGUS_ANGLE_VALUE = -400
    FRONT_VAL = 0
    LEFT_VAL = -1
    RIGHT_VAL = 1

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.lt, self.rt = self.__load_best_thresholds()
        img_filters = self.__load_img_filters()
        self.keep_roi_classes = img_filters[sf_constants.FILTER_KEEP_CLASSES]
        self.roi_area = img_filters[sf_constants.FILTER_AREA_VALUE]

    # not too happy about this code duplication from roi_classif_mq_predictor - perhaps creating another abstract class,
    #  that includes these 2 methods and the stuff from utils which should be extended by both predictors...
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

    def __load_img_filters(self):
        """ Loads the filters used at training time which contain the list of ROI classes we do predictions on and the
         image area threshold. """

        path = os.path.join(self.get_alg_bundle_path(self.ALG_NAME), "model_img_filters.json")
        return io_utils.json_load(path)

    def __load_best_thresholds(self):
        """ Loads the best thresholds computed for the current model, these will be used to update predictions, by
        changing the prediction value based on the threshold value. """

        path = os.path.join(self.get_alg_bundle_path(self.ALG_NAME), "model_best_thresholds.json")
        return postprocess.load_best_thresholds_json(path)

    def __get_class_2_conf_dict(self, prediction):
        class_conf = {}

        for index, confidence in enumerate(prediction):
            class_name = self.index_2_params[0].classIndex_2_class[index]
            class_conf[class_name] = confidence

        return class_conf

    def __get_pred_value_with_thresholds(self, prediction):
        """ Applies the best confidence thresholds over the prediction and returns the value to be saved in angle_of_roi.
         This angle_of_roi value is an integer of -1 for left orientation, 0 for front and 1 for right. This way, we can
         update with a real angle value based on our tests (this may be 90 degrees, or we may decide that 60 degrees is a
          better value). """

        logger.debug('updating confidence based on thresholds lt: {} and rt: {} ...'.format(self.lt, self.rt))
        logger.debug('confidence before threshold: {}'.format(prediction))

        class_2_conf = self.__get_class_2_conf_dict(prediction)
        front_conf = class_2_conf[SignFacingLabel.CLS_FRONT]
        left_conf = class_2_conf[SignFacingLabel.CLS_LEFT]
        right_conf = class_2_conf[SignFacingLabel.CLS_RIGHT]

        value = self.FRONT_VAL
        if front_conf > left_conf and front_conf > right_conf:
            value = self.FRONT_VAL
        elif left_conf > self.lt and right_conf > self.rt:
            value = self.LEFT_VAL if left_conf >= right_conf else self.RIGHT_VAL
        elif left_conf > self.lt:
            value = self.LEFT_VAL
        elif right_conf > self.rt:
            value = self.RIGHT_VAL

        logger.debug('the resulting value: {}'.format(value))

        return value

    def __should_run_prediction(self, roi):
        """ Returns true if we should run a prediction on the provided roi, false if not. """

        run_prediction = proto_api.get_roi_type_name(roi.type) in self.keep_roi_classes and\
                         utils.compute_roi_area(roi.rect.br.col, roi.rect.tl.col, roi.rect.br.row, roi.rect.tl.row) >= self.roi_area
        logger.debug('should run prediction on roi_id {}: {}'.format(utils.roi_2_id(roi), run_prediction))

        return run_prediction

    def __update_angle_of_roi_inplace(self, roi_proto, value):
        """ Updates the roi proto, setting a value in the local field for the angle_of_roi. This value is not an actual
         angle but can be -1 for left oriented signs, 1 for right oriented signs, and 0 for front oriented signs. """

        logger.debug('updating angle of roi_id {} - value {} '.format(utils.roi_2_id(roi_proto), value))
        proto_local = proto_api.get_new_processed_sign_localization_proto(self.BOGUS_ANGLE_VALUE,
                                                                          self.BOGUS_ANGLE_VALUE, 0,
                                                                          self.BOGUS_ANGLE_VALUE,
                                                                          self.BOGUS_ANGLE_VALUE, value)
        roi_proto.local.CopyFrom(proto_local)

    def __update_results_in_proto(self, predictions_2_ids, image_proto):
        """ If the rois in the image_proto are available in predictions_2_ids it means we have a prediction for
         the respective roi and we will apply best threhsolds on that. If the roi isn't available we automatically set
         the result class to 'front', meaning the angle of roi will be 0. """

        pred_dict = {roi_id: pred for pred, roi_id in predictions_2_ids}
        image_rois = image_proto.rois

        for roi_proto in image_rois:
            roi_id = utils.roi_2_id(roi_proto)

            if roi_id in pred_dict:
                # apply best threhsolds to get value, then update roi_proto inplace
                self.__update_angle_of_roi_inplace(roi_proto, self.__get_pred_value_with_thresholds(pred_dict[roi_id]))
                logger.debug('update results in proto - we have a prediction {} for roi_id {}'.format(pred_dict[roi_id],
                                                                                                     roi_id))
            else:
                self.__update_angle_of_roi_inplace(roi_proto, 0)  # set all rois without predictions to front.
                logger.debug('update results in proto - no prediction for roi_id {}'.format(roi_id))

    def __get_class_name_with_confidence(self, pred, index):
        return "{}:{}".format(self.index_2_params[0].classIndex_2_class[index], round(float(pred[index]), 4))

    def __audit_predictions(self, image_proto, predictions_2_ids):

        pred_2_ids_str = ["{},{},{},{}".format(roi_id,
                                            self.__get_class_name_with_confidence(pred, 0),
                                            self.__get_class_name_with_confidence(pred, 1),
                                            self.__get_class_name_with_confidence(pred, 2))
                          for pred, roi_id in predictions_2_ids]

        super().set_audit_key_val(image_proto.metadata.image_path, self.ES_KEY_ROI_FACING, pred_2_ids_str)

    def preprocess(self, image_proto):
        if image_proto.rois is None or len(image_proto.rois) == 0:
            return list()

        img = self.read_image(image_proto)
        rois_proto = [roi for roi in image_proto.rois if self.__should_run_prediction(roi)]
        rois_2_ids = [self.__resize_cropped_roi(img, roi_proto) for roi_proto in rois_proto]

        return rois_2_ids

    def predict(self, rois_2_ids_lists):
        if self.model is None:
            self.model = self.load_model()

        predictions_2_ids_list = [self.__predict_batch(rois_2_ids) for rois_2_ids in rois_2_ids_lists]
        logger.debug("predict - predictions: {}".format(predictions_2_ids_list))

        return predictions_2_ids_list

    def postprocess(self, predictions_2_ids, image_proto):
        self.__update_results_in_proto(predictions_2_ids, image_proto)
        self.__audit_predictions(image_proto, predictions_2_ids)

        logger.debug("resulting proto {}".format(image_proto))

        return image_proto


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="config file path", type=str, required=True)

    return parser.parse_args()


def run_predictor(conf_file):
    conf = io_utils.config_load(conf_file)
    predictor = SignsFacingClassifPredictor(conf)
    predictor.start()


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    args = parse_arguments()
    run_predictor(args.config_file)
