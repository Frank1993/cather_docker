
class HazardType:
    SPEED_BUMP = "speed_bump"
    BIG_POTHOLE = "big_pothole"
    SMALL_POTHOLE = "small_pothole"
    SEWER_HOLE = "sewer_hole"
    CLEAR = "clear"
    
    @staticmethod
    def get_all():
        return [HazardType.SPEED_BUMP, HazardType.BIG_POTHOLE, HazardType.SMALL_POTHOLE, HazardType.SEWER_HOLE, HazardType.CLEAR]
        
    
class ConfigParams:
    DATASET_BASE_PATH = "dataset_base_path"
    FEATURES = "features_cols"
    FREQUENCY = "frequency"
    DRIVE_FOLDERS = "drive_folders"
    SPECIFIC_HAZARDS = "specific_hazards"
    BLACKLIST = "blacklist_trips"
    STEPS = "steps"
    SCALER_TYPE = 'scaler'
    SUFFIX = "suffix"
    PHONE_NAME = "phone_name"
    DERIVED_WINDOW_SIZE = "derived_window_size"
    HAZARD_BUFFER_STEPS = "hazard_buffer"
    CLASS_2_INDEX = "class_2_index"
    INDEX_2_CLASS = "index_2_class"
    DATASET_CONFIG_PATH = "dataset_config_path"
    TEST_DRIVE_DAY = "test_drive_day"
    KEPT_HAZARDS = "kept_hazards"
    TRAIN_CLASS_BALANCE_FACTOR = "train_class_balance_factor"
    CKPT_FOLDER = "ckpt_folder"
    BATCH_SIZE = "batch_size"
    EPOCHS = "epochs"
    CONF_THRESHOLD = "conf_threshold"
    DBSCAN_EPS = "dbscan_eps"
    DBSCAN_MIN_SAMPLES = "dbscan_min_samples"
    BUNDLE_PATH = "bundle_path"
    PREDICT_INPUT_FOLDER = "pred_input_folder"
    PREDICT_OUTPUT_FOLDER = "pred_output_folder"
    SINGLE_TRIP_FILTER = "single_trip_filter"
    WITH_META_AGGREGATION = "with_meta_aggregation"
    WITH_META_MIN_TRIPS_FILTERING = "with_meta_min_trips_filtering"
    
class Column:
    DATETIME = "datetime"
    RAW_HAZARD = "raw_hazard"
    HAZARD_LAT = "hazard_lat"
    HAZARD_LON = "hazard_lon"
    LOCATION = "location"
    TIMESTAMP = "timestamp"
    LAT = "lat"
    LON = "lon"
    ALL_FEATURES = "all_features"
    HAZARD = "hazard"
    TRIP_ID = "trip_id"
    PRED = "pred"
    IMAGE_INDEX = "tFIndex"
    CLUSTER_ID = 'cluster_id'
    CLUSTER_LAT = 'cluster_lat'
    CLUSTER_LON = 'cluster_lon'
    MEMBER_IDS = 'member_ids'
    META_CLUSTER_ID = 'meta_cluster_id'
    META_CLUSTER_LAT = 'meta_cluster_lat'
    META_CLUSTER_LON = 'meta_cluster_lon'
    HAS_ENOUGH_TRIPS = "has_enough_trips"

    
class FolderName:
    OSC_SENSOR_DATA = "osc_sensor_data"
    CONFIG = "config"
    HAZARD_DATA = "hazard_data"
    