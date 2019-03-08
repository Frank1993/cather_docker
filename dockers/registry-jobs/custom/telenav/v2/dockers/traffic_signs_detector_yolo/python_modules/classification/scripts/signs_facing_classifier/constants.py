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
