# threshold values
THRESHOLD_NAME_LEFT = "lt"
THRESHOLD_NAME_RIGHT = "rt"
MODEL_SCORE = "score"

# filter values
FILTER_KEEP_CLASSES = "keep_classes"
FILTER_AREA_VALUE = "area_value"

# file names we have as output
MODEL_FILTERS = "model_img_filters.json"
BEST_THREHSOLDS = "model_best_thresholds.csv"  # TODO maybe save these as json, since we only save one value for each

# ROI filename related stuff
FILENAME_SPLIT_REGEX = "_[0-9]+_[0-9]+_[0-9]+_[0-9]+_[A-Z]"
IMAGE_EXTENSION = ".jpg"


# clasification class names
class SignFacingLabel:
    CLS_FRONT = "front"
    CLS_LEFT = "left"
    CLS_RIGHT = "right"


# prediction dataframe post processing columns
class PredDfColumn:
    CROP_NAME_COL = 'crop_name'
    CONF_FRONT_COL = 'conf_front'
    CONF_LEFT_COL = 'conf_left'
    CONF_RIGHT_COL = 'conf_right'
    GT_CLASS_COL = 'label_class'
    PRED_CLASS_COL = 'pred_class'


# ROI dataframe for pre processing (matching, filtering, data augmentation, etc)
class RoiDfColumn:
    MATCHED_COL = 'matched_coords'
    IS_MATCHED_COL = 'is_matched'
    CROP_PATH_COL = 'crop_path'
    IMG_NAME_COL = 'image'
    TL_ROW_COL = 'tl_row'
    TL_COL_COL = 'tl_col'
    BR_ROW_COL = 'br_row'
    BR_COL_COL = 'br_col'
    ROI_CLASS_COL = 'roi_class'
    ORIENTATION_COL = 'roi_orientation'
    ROI_AREA_COL = 'roi_area'
