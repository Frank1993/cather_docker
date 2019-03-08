class HazardType:
    SPEED_BUMP = "speed_bump"
    BIG_POTHOLE = "big_pothole"
    SMALL_POTHOLE = "small_pothole"
    SEWER_HOLE = "sewer_hole"
    BUMPY_ROAD_START = "bumpy_road_start"
    BUMPY_ROAD_END = "bumpy_road_end"
    BUMPY_ROAD = "bumpy_road"
    CLEAR = "clear"
    UNPAVED_ROAD_START = "unpaved_road_start"
    UNPAVED_ROAD_END = "unpaved_road_end"
    UNPAVED_ROAD = "unpaved_road"

    @staticmethod
    def get_all():
        return [HazardType.SPEED_BUMP, HazardType.BIG_POTHOLE, HazardType.SMALL_POTHOLE, HazardType.SEWER_HOLE,
                HazardType.BUMPY_ROAD_START, HazardType.BUMPY_ROAD_END, HazardType.BUMPY_ROAD,
                HazardType.UNPAVED_ROAD_START,
                HazardType.UNPAVED_ROAD_END, HazardType.UNPAVED_ROAD, HazardType.CLEAR]


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
    TEST_DRIVE_DAYS = "test_drive_days"
    KEPT_HAZARDS = "kept_hazards"
    TRAIN_CLASS_BALANCE_FACTOR = "train_class_balance_factor"
    CKPT_FOLDER = "ckpt_folder"
    BATCH_SIZE = "batch_size"
    EPOCHS = "epochs"
    CONF_THRESHOLD = "conf_threshold"
    DBSCAN_EPS = "dbscan_eps"
    DBSCAN_MIN_SAMPLES = "dbscan_min_samples"
    BUNDLE_PATH = "bundle_path"
    PREDICT_INPUT_FOLDERS = "pred_input_folders"
    PREDICT_OUTPUT_FOLDER = "pred_output_folder"
    SINGLE_TRIP_FILTER = "single_trip_filter"
    WITH_META_AGGREGATION = "with_meta_aggregation"
    WITH_META_MIN_TRIPS_FILTERING = "with_meta_min_trips_filtering"
    CROP_START = "crop_start"
    ADD_MATCH_DATA = "add_match_data"
    WITH_CUSTOM_WAY_SECTIONS = "with_custom_way_sections"
    WITH_EVALUATE = "with_evaluate"


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
    FORWARD = "forward"
    WAY_ID = "way_id"
    NODE_ID = "node_id"
    FROM_NODE_ID = "from_node_id"
    TO_NODE_ID = "to_node_id"
    MATCHED_LAT = "matched_lat"
    MATCHED_LON = "matched_lon"
    COMPUTED_HEADING = "computed_heading"
    OFFSET = "offset"
    GT_SCORE = "gt_score"
    PRED_SCORE = "pred_score"
    PRED_DIFF = "pred_diff"
    GT_ROAD_CLASS = "gt_road_class"
    PRED_ROAD_CLASS = "pred_road_class"
    ROAD_CLASS_DIFF = "road_class_diff"
    WAY_LENGTH = "way_length"
    WAY_LENGTH_BUCKET = "way_length_bucket"
    IS_SAME_ROAD_CLASS = "is_same_road_class"
    SECTION_FROM_NODE_ID = "section_from_node_id"
    SECTION_FROM_NODE_ID_LAT = "section_from_node_id_lat"
    SECTION_FROM_NODE_ID_LON = "section_from_node_id_lon"
    SECTION_TO_NODE_ID = "section_to_node_id"
    SECTION_TO_NODE_ID_LAT = "section_to_node_id_lat"
    SECTION_TO_NODE_ID_LON = "section_to_node_id_lon"
    COUNT = "count"
    PRED_PROBA = "pred_proba"
    PHONE_TYPE = "phone_type"
    IS_SAME_CLASS = "is_same_class"
    PRED_LIST = "pred_list"
    FROM_NEIGHBOUR_ID = "from_neighbour_id"
    FROM_NEIGHBOUR_PRED_CLASS = "from_neighbour_pred_class"
    TO_NEIGHBOUR_ID = "to_neighbour_id"
    TO_NEIGHBOUR_PRED_CLASS = "to_neighbour_pred_class"
    TRIP_LIST = "trip_list"
    NR_UNIQUE_TRIPS = "nr_unique_trips"
    WITH_CUSTOM_WAY_SECTIONS = "with_custom_way_sections"


class FolderName:
    OSC_SENSOR_DATA = "osc_sensor_data"
    CONFIG = "config"
    HAZARD_DATA = "hazard_data"
