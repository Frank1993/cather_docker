import argparse
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import apollo_python_common.proto_api as proto_api
from sign_positioning.random_forrest.regressor_feature_data import FeatureData
from sign_positioning.mq.position_calculator import PositionCalculator
import sign_positioning.random_forrest.constants as rf_constants


class SignPositioningRF:

    def __init__(self, train_predicted_rois_path, train_ground_truth_rois_path, test_rois_path,
                 test_output_path, phone_lenses_path,
                 roi_dimensions_path, with_angle):
        self.feature_data = FeatureData(train_predicted_rois_path, train_ground_truth_rois_path, phone_lenses_path,
                                       roi_dimensions_path)
        self.test_rois_path = test_rois_path
        self.test_output_path = test_output_path
        self.with_angle = with_angle

    @staticmethod
    def _get_data(train_df, test_df, feature_cols, label_col):
        x_train, y_train = train_df[feature_cols].values, train_df[label_col]
        x_test, y_test = test_df[feature_cols].values, test_df[label_col]
        return x_train, y_train, x_test, y_test

    @staticmethod
    def _grid_search(x_train, y_train):
        param_grid = {
                         'bootstrap': [True, False],
                         'max_depth': [1, 2, 3, 4, 5, 10, None],
                         'max_features': ['auto', 'sqrt'],
                         'min_samples_leaf': [1, 2, 3, 4, 5],
                         'min_samples_split': [2, 3, 4],
                         'n_estimators': [200, 400, 500, 600, 700, 800, 900, 1000]
                     }
        rf = RandomForestRegressor()
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(x_train, y_train)
        return grid_search.best_params_

    @staticmethod
    def _train_rf_model(x_train, y_train, params):
        rf = RandomForestRegressor(
            bootstrap=params['bootstrap'],
            max_depth=params['max_depth'],
            max_features=params['max_features'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            n_estimators=params['n_estimators']
        )
        rf.fit(x_train, y_train)
        return rf

    def _split_train_test_by_trip_id(self, df, train_percentage):
        id_col = rf_constants.TRIP_ID_STR
        trip_ids = df[id_col].unique()
        nr_train_ids = max(1, int(len(trip_ids) * train_percentage))
        train_ids = trip_ids[:nr_train_ids]
        test_ids = trip_ids[nr_train_ids:]
        train_df = df.loc[df[id_col].isin(train_ids)].sample(frac=1)
        test_df = df.loc[df[id_col].isin(test_ids)].sample(frac=1)
        return train_df, test_df

    def _train_validate_model(self, features_col, label_col, pred_label, train_df, test_df):
        x_train, y_train, x_test, y_test = self._get_data(train_df, test_df,
                                                         features_col,
                                                         label_col)
        best_params = self._grid_search(x_train, y_train)
        rf = self._train_rf_model(x_train, y_train, best_params)
        initial_mae = mean_absolute_error(test_df[pred_label], test_df[label_col])
        y_pred = rf.predict(x_test)
        rf_mae = mean_absolute_error(y_test, y_pred)
        print("{} {} from {} to {} mean absolute error".format(label_col, pred_label, initial_mae, rf_mae))
        return rf

    def train_distance_angle_models(self):
        df = self.feature_data.get_df()
        train_df, test_df = self._split_train_test_by_trip_id(df, 0.7)
        rf_distance = self._train_validate_model(rf_constants.FEATURES_LIST,
                                                 rf_constants.DISTANCE_LABEL_STR, rf_constants.ROI_PRED_DISTANCE_STR, train_df,
                                                 test_df)
        rf_angle = None
        if self.with_angle:
            rf_angle = self._train_validate_model(rf_constants.FEATURES_LIST,
                                                  rf_constants.ANGLE_LABEL_STR, rf_constants.ROI_PRED_SIGN_HEADING_STR, train_df,
                                                  test_df)
        return rf_distance, rf_angle

    def predict(self, rf_distance, rf_angle):
        rois_metadata = proto_api.read_imageset_file(self.test_rois_path)
        for image in rois_metadata.images:
            for roi in image.rois:
                features_dict = self.feature_data.initialize_features_data()
                roi_features = self.feature_data.get_features_from_roi_data(roi, image)
                for key, value in roi_features.items():
                    features_dict[key].append(value)
                df_features = pd.DataFrame(features_dict)
                pred_distance = rf_distance.predict(df_features[rf_constants.FEATURES_LIST].values)
                if self.with_angle:
                    pred_angle = rf_angle.predict(df_features[rf_constants.FEATURES_LIST].values)
                else:
                    pred_angle = image.match_data.matched_heading + roi.local.angle_from_center
                lat = image.match_data.matched_position.latitude
                lon = image.match_data.matched_position.longitude
                sign_lat, sign_lon = PositionCalculator.calculate_position_offset_in_given_direction(lat, lon,
                                                                                                     pred_angle,
                                                                                                     pred_distance)
                roi.local.position.latitude = sign_lat
                roi.local.position.longitude = sign_lon
        proto_api.serialize_proto_instance(rois_metadata, self.test_output_path, "regressor_output")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--predicted_rois_path",
                        type=str, required=True)
    parser.add_argument("-e", "--ground_truth_rois_path",
                        type=str, required=True)
    parser.add_argument("-t", "--test_rois_path",
                        type=str, required=True)
    parser.add_argument("-o", "--test_output_path",
                        type=str, required=True)
    parser.add_argument("-p", "--phone_lenses_path",
                        type=str, required=True)
    parser.add_argument("-r", "--rois_dimension_path",
                        type=str, required=True)
    parser.add_argument("-y", "--with_angle",
                        action='store_true')

    args = parser.parse_args()

    sign_positioner_rf = SignPositioningRF(args.predicted_rois_path, args.ground_truth_rois_path,
                                           args.test_rois_path, args.test_output_path, args.phone_lenses_path,
                                           args.rois_dimension_path, args.with_angle)
    rf_distance, rf_angle = sign_positioner_rf.train_distance_angle_models()
    sign_positioner_rf.predict(rf_distance, rf_angle)
