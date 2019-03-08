import numpy as np
import pandas as pd

import classification.scripts.signs_facing_classifier.constants as sf_constants
from classification.scripts.signs_facing_classifier.constants import SignFacingLabel
from classification.scripts.signs_facing_classifier.constants import PredDfColumn
from classification.scripts.constants import Column
from apollo_python_common import io_utils


def compute_model_score(pred_data_df):
    """ Computes the score of the trained model having the predictions data frame as an input. The metric is computed by
    subtracting the misclassified front images count from the sum of correctly classified left and right images. """

    mc_front = pred_data_df.loc[(pred_data_df[Column.PRED_CLASS_COL] != SignFacingLabel.CLS_FRONT)
                                & (pred_data_df[Column.LABEL_CLASS_COL] == SignFacingLabel.CLS_FRONT)].shape[0]
    ok_left = pred_data_df.loc[(pred_data_df[Column.PRED_CLASS_COL] == SignFacingLabel.CLS_LEFT)
                               & (pred_data_df[Column.LABEL_CLASS_COL] == SignFacingLabel.CLS_LEFT)].shape[0]
    ok_right = pred_data_df.loc[(pred_data_df[Column.PRED_CLASS_COL] == SignFacingLabel.CLS_RIGHT)
                                & (pred_data_df[Column.LABEL_CLASS_COL] == SignFacingLabel.CLS_RIGHT)].shape[0]
    score = ok_right + ok_left - mc_front

    return score


def get_pred_confidence_df(prediction_df, params):
    """ Given a predictions data frame and the predictor params, returns a dataframe containing the confidence
     for each predicted class as a separate column, as well as the name of the image crop for which the prediction
     was run and the predicted class and ground truth class for each prediction. """

    prediction_df[PredDfColumn.CROP_NAME_COL] = prediction_df.index
    prediction_df[PredDfColumn.CONF_FRONT_COL] = prediction_df[Column.PRED_COL].apply(
        lambda arr: arr[params.class_2_classIndex[SignFacingLabel.CLS_FRONT]])
    prediction_df[PredDfColumn.CONF_LEFT_COL] = prediction_df[Column.PRED_COL].apply(
        lambda arr: arr[params.class_2_classIndex[SignFacingLabel.CLS_LEFT]])
    prediction_df[PredDfColumn.CONF_RIGHT_COL] = prediction_df[Column.PRED_COL].apply(
        lambda arr: arr[params.class_2_classIndex[SignFacingLabel.CLS_RIGHT]])

    pred_df = prediction_df.filter(items=[PredDfColumn.CROP_NAME_COL, PredDfColumn.CONF_FRONT_COL, PredDfColumn.CONF_LEFT_COL,
                                          PredDfColumn.CONF_RIGHT_COL, PredDfColumn.GT_CLASS_COL, PredDfColumn.PRED_CLASS_COL])

    return pred_df


def get_class_for_thresh(r, lt, rt):
    """ Computes and returns the sign facing label (front, left or right) after applying the best left and right
     thresholds on one prediction dataframe row. """

    front_conf = r[PredDfColumn.CONF_FRONT_COL]
    left_conf = r[PredDfColumn.CONF_LEFT_COL]
    right_conf = r[PredDfColumn.CONF_RIGHT_COL]

    if front_conf > left_conf and front_conf > right_conf:
        return SignFacingLabel.CLS_FRONT
    elif left_conf > lt and right_conf > rt:
        return SignFacingLabel.CLS_LEFT if left_conf >= right_conf else SignFacingLabel.CLS_RIGHT
    elif left_conf > lt:
        return SignFacingLabel.CLS_LEFT
    elif right_conf > rt:
        return SignFacingLabel.CLS_RIGHT


def apply_thresholds(pred_df, lt, rt):
    """ Applies the left and right best thresholds over the predictions data frame given as an argument. """

    pred_df.loc[:, PredDfColumn.PRED_CLASS_COL] = pred_df.apply(lambda r: get_class_for_thresh(r, lt, rt), axis=1)

    return pred_df


def compute_best_thresholds(pred_df):
    """ The method used for computing the best confidence thresholds for the left and right predicted classes. It runs
     over a range of values from 0.01 to 1.00 with a step of 0.01 for both left and right thresholds and computes the
     best model score for the given left and right threshold pair. It then applies a max over the model score and returns
     a dataframe containing only the threshold pairs that have the max score. """

    thresholds = {sf_constants.MODEL_SCORE: [], sf_constants.THRESHOLD_NAME_LEFT: [], sf_constants.THRESHOLD_NAME_RIGHT: []}
    for lt in np.arange(0.1, 1, 0.01):
        for rt in np.arange(0.1, 1, 0.01):
            thresh_pred_df = apply_thresholds(pred_df, lt, rt)
            thresholds[sf_constants.THRESHOLD_NAME_LEFT].append(lt)
            thresholds[sf_constants.THRESHOLD_NAME_RIGHT].append(rt)
            thresholds[sf_constants.MODEL_SCORE].append(compute_model_score(thresh_pred_df))

    thresholds_df = pd.DataFrame(thresholds)
    max_score = max(thresholds_df[sf_constants.MODEL_SCORE])

    best_thresholds_df = thresholds_df[thresholds_df[sf_constants.MODEL_SCORE] == max_score] \
        .sort_values([sf_constants.THRESHOLD_NAME_LEFT, sf_constants.THRESHOLD_NAME_RIGHT])
    best_thresholds_df = best_thresholds_df.reset_index(drop=True)

    return best_thresholds_df


def load_best_thresholds_json(path):
    """ Loads the left and right threshold values from the json file given as argument. """

    best_thresholds = io_utils.json_load(path)

    return best_thresholds[sf_constants.THRESHOLD_NAME_LEFT], best_thresholds[sf_constants.THRESHOLD_NAME_RIGHT]