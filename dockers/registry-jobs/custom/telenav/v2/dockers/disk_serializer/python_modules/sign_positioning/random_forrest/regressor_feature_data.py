import pandas as pd

import apollo_python_common.proto_api as proto_api
import apollo_python_common.map_geometry.geometry_utils as geometry_utils

import sign_positioning.random_forrest.constants as rf_constants

class FeatureData:

    def __init__(self, predicted_rois_path, ground_truth_rois_path, phone_lenses_path,
                 roi_dimensions_path):
        self.features_data = self.initialize_features_data()
        self.predicted_metadata = proto_api.read_imageset_file(predicted_rois_path)
        self.ground_truth_metadata = proto_api.read_imageset_file(ground_truth_rois_path)
        self.signs_dimensions = proto_api.create_sign_dimensions_dict(
            proto_api.read_sign_dimensions(roi_dimensions_path))
        self.phone_lenses = proto_api.create_phone_lenses_dict(proto_api.read_phone_lenses(phone_lenses_path))

    def get_df(self):
        self._add_feature_data()
        return pd.DataFrame(self.features_data)

    def add_image_data(self, image, predicted_roi, ground_truth_roi):
        start_lat = image.match_data.matched_position.latitude
        start_lon = image.match_data.matched_position.longitude
        end_lat = ground_truth_roi.local.position.latitude
        end_lon = ground_truth_roi.local.position.longitude
        gt_roi_distance = geometry_utils.compute_haversine_distance(start_lon, start_lat,
                                                                    end_lon, end_lat)
        gt_roi_heading = geometry_utils.compute_heading(start_lat, start_lon, end_lat, end_lon)
        self._add_roi_data(predicted_roi, image, gt_roi_distance, gt_roi_heading)

    @staticmethod
    def initialize_features_data():
        features_data = dict()
        features_data[rf_constants.TRIP_ID_STR] = list()
        for feature in rf_constants.FEATURES_LIST:
            features_data[feature] = list()
        features_data[rf_constants.DISTANCE_LABEL_STR] = list()
        features_data[rf_constants.ANGLE_LABEL_STR] = list()
        return features_data

    def _add_roi_data_to_features(self, predicted_roi, image, gt_roi_distance, gt_roi_heading):
        roi_features = self.get_features_from_roi_data(predicted_roi, image)
        roi_features[rf_constants.DISTANCE_LABEL_STR] = gt_roi_distance
        roi_features[rf_constants.ANGLE_LABEL_STR] = gt_roi_heading
        for key, value in roi_features.items():
            self.features_data[key].append(value)

    def get_features_from_roi_data(self, predicted_roi, image):
        features = self.initialize_features_data()
        features[rf_constants.TRIP_ID_STR] = image.metadata.trip_id
        features[rf_constants.IMAGE_RES_WIDTH_STR] = image.sensor_data.img_res.width
        features[rf_constants.IMAGE_RES_HEIGHT_STR] = image.sensor_data.img_res.height
        features[rf_constants.IMAGE_SPEED_STR] = image.sensor_data.speed_kmh
        features[rf_constants.IMAGE_GPS_ACCURACY_STR] = image.sensor_data.gps_accuracy
        phone_lens = self.phone_lenses[image.sensor_data.device_type]
        features[rf_constants.FOCAL_LENGTH_STR] = phone_lens.focal_length
        features[rf_constants.PIXEL_WIDTH_STR] = phone_lens.pixel_width
        features[rf_constants.PIXEL_HEIGHT_STR] = phone_lens.pixel_height
        features[rf_constants.SENSOR_WIDTH_STR] = phone_lens.sensor_width
        features[rf_constants.SENSOR_HEIGHT_STR] = phone_lens.sensor_height
        if image.features.vanishing_point.vp.col > 0 and image.features.vanishing_point.vp.row > 0:
            features[rf_constants.VP_X_PERCENTAGE_STR] = image.features.vanishing_point.vp.col / image.sensor_data.img_res.width
            features[rf_constants.VP_Y_PERCENTAGE_STR] = image.features.vanishing_point.vp.row / image.sensor_data.img_res.height
        else:
            features[rf_constants.VP_X_PERCENTAGE_STR] = 0.5
            features[rf_constants.VP_Y_PERCENTAGE_STR] = 0.5
        features[rf_constants.VP_CONFIDENCE_STR] = image.features.vanishing_point.confidence
        obj_real_dimensions = self.signs_dimensions[predicted_roi.type]
        features[rf_constants.SIGN_HEIGHT_STR] = obj_real_dimensions.height
        features[rf_constants.SIGN_WIDTH_STR] = obj_real_dimensions.width

        features[rf_constants.ROI_X_PERCENTAGE_STR] = predicted_roi.rect.tl.col / image.sensor_data.img_res.width
        features[rf_constants.ROI_Y_PERCENTAGE_STR] = predicted_roi.rect.tl.row / image.sensor_data.img_res.height

        features[rf_constants.ROI_WIDTH_STR] = predicted_roi.rect.br.col - predicted_roi.rect.tl.col
        features[rf_constants.ROI_HEIGHT_STR] = predicted_roi.rect.br.row - predicted_roi.rect.tl.row

        features[rf_constants.ROI_WIDTH_PERCENTAGE_STR] = (predicted_roi.rect.br.col - predicted_roi.rect.tl.col) / image.sensor_data.img_res.width
        features[rf_constants.ROI_HEIGHT_PERCENTAGE_STR] = (predicted_roi.rect.br.row - predicted_roi.rect.tl.row) / image.sensor_data.img_res.height

        features[rf_constants.ROI_CONFIDENCE_STR] = predicted_roi.detections[0].confidence
        features[rf_constants.ROI_ANGLE_OF_ROI_STR] = predicted_roi.local.angle_of_roi
        features[rf_constants.ROI_ANGLE_FROM_CENTER_STR] = predicted_roi.local.angle_from_center
        features[rf_constants.ROI_PRED_DISTANCE_STR] = predicted_roi.local.distance / 1000
        features[rf_constants.ROI_PRED_SIGN_HEADING_STR] = predicted_roi.local.angle_from_center + image.match_data.matched_heading
        return features

    @staticmethod
    def _image_proto_equality(image_a, image_b):
        trip_id_a = int(float(image_a.metadata.trip_id))
        trip_id_b = int(float(image_b.metadata.trip_id))
        image_index_a = image_a.metadata.image_index
        image_index_b = image_b.metadata.image_index
        return trip_id_a == trip_id_b and image_index_a == image_index_b

    def _find_equal_image(self, searched_image, images):
        for image in images:
            if self._image_proto_equality(searched_image, image):
                return image

    @staticmethod
    def _find_equal_roi(searched_roi, rois):
        for roi in rois:
            if roi.id == searched_roi.id:
                return roi

    def _add_feature_data(self):
        for predicted_image in self.predicted_metadata.images:
            ground_truth_image = self._find_equal_image(predicted_image, self.ground_truth_metadata.images)
            if not ground_truth_image:
                continue
            for predicted_roi in predicted_image.rois:
                ground_truth_roi = self._find_equal_roi(predicted_roi, ground_truth_image.rois)
                if not ground_truth_roi:
                    continue
                start_lat = predicted_image.match_data.matched_position.latitude
                start_lon = predicted_image.match_data.matched_position.longitude
                end_lat = ground_truth_roi.local.position.latitude
                end_lon = ground_truth_roi.local.position.longitude
                gt_roi_distance = geometry_utils.compute_haversine_distance(start_lon, start_lat,
                                                                            end_lon, end_lat)
                gt_roi_heading = geometry_utils.compute_heading(start_lat, start_lon, end_lat, end_lon)
                self._add_roi_data_to_features(predicted_roi, predicted_image, gt_roi_distance, gt_roi_heading)
                continue