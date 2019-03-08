import os

import numpy as np
from fastai.vision import ClassificationInterpretation

from classification.fast_ai.signs_facing_classifier.utils.constants import PredDfColumn, SignFacingLabel, RoiDfColumn


def compute_model_score(pred_df):
    """ Computes the score of the trained model having the predictions data frame as an input. The metric is computed by
    subtracting the misclassified front images count from the sum of correctly classified left and right images. """

    mc_front = pred_df.loc[(pred_df[PredDfColumn.PRED_CLASS_COL] != SignFacingLabel.FRONT)
                           & (pred_df[RoiDfColumn.ORIENTATION_COL] == SignFacingLabel.FRONT)].shape[0]
    ok_left = pred_df.loc[(pred_df[PredDfColumn.PRED_CLASS_COL] == SignFacingLabel.LEFT)
                          & (pred_df[RoiDfColumn.ORIENTATION_COL] == SignFacingLabel.LEFT)].shape[0]
    ok_right = pred_df.loc[(pred_df[PredDfColumn.PRED_CLASS_COL] == SignFacingLabel.RIGHT)
                           & (pred_df[RoiDfColumn.ORIENTATION_COL] == SignFacingLabel.RIGHT)].shape[0]
    print("mc_front: {} - ok_left: {} - ok_right: {}".format(mc_front, ok_left, ok_right))
    score = ok_right + ok_left - mc_front

    return score


def get_img_path(row, path):
    """ Based on a given roi dataframe row and a source dir path, returns the corresponding crop file path. """
    coords = row[RoiDfColumn.TL_COL_COL], row[RoiDfColumn.TL_ROW_COL], row[RoiDfColumn.BR_COL_COL], \
             row[RoiDfColumn.BR_ROW_COL]

    crop_name = row[RoiDfColumn.IMG_NAME_COL]
    for coord in coords:
        crop_name = crop_name + '_' + str(coord)

    crop_path = os.path.join(path, row[RoiDfColumn.ORIENTATION_COL])

    return os.path.join(crop_path, '{}_{}.jpg'.format(crop_name, row[RoiDfColumn.ROI_CLASS_COL]))


def model_stats(learner, tta=False):
    """ Returns the model precision and the confusion matrix. """

    interp = ClassificationInterpretation.from_learner(learner, tta=tta)
    precision = np.trace(interp.confusion_matrix()) / interp.confusion_matrix().sum() * 100
    confusion_matrix = interp.confusion_matrix()

    return precision, confusion_matrix
