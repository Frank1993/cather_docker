import os


class MQ_Param:
    COMPONENT = "component"
    ALGORITHM = "algorithm"
    MQ_HOST = "mq_host"
    MQ_PORT = "mq_port"
    MQ_USERNAME = "mq_username"
    MQ_PASSWORD = "mq_password"
    MQ_INPUT_QUEUE_NAME = "mq_input_queue_name"
    MQ_INPUT_ERRORS_QUEUE_NAME = "mq_input_errors_queue_name"
    MQ_OUTPUT_QUEUE_NAME = "mq_output_queue_name"
    MAX_INTERNAL_QUEUE_SIZE = "max_internal_queue_size"
    PREDICT_BATCH_SIZE = "predict_batch_size"
    ELASTICSEARCH_HOST = "elasticsearch_host"
    ELASTICSEARCH_AUDIT_INDEX_NAME = "elasticsearch_audit_index_name"
    ELASTICSEARCH_ENABLED = "elasticsearch_enabled"
    NO_ACK = "no_ack"
    MQ_PREFETCH_COUNT = "mq_prefetch_count"
    MQ_NR_PREPROCESS_THREADS = "nr_preprocess_threads"
    MQ_NR_PREDICT_THREADS = "nr_predict_threads"
    LOGSTASH_HOST = "logstash_host"
    LOGSTASH_PORT = "logstash_port"
    PRED_THRESHOLDS = "pred_thresholds"
    PER_PROCESS_GPU_MEMORY_FRACTION = "per_process_gpu_memory_fraction"
    ALLOW_GROWTH_GPU_MEMORY = "allow_growth_gpu_memory"
    ALGORITHMS = "algorithms"
    ALGORITHM_VERSION = "algorithm_version"
    BUNDLE_PATH = "bundle_path"
    HIGH_QUALITY_QUEUE_NAME = "high_quality_queue_name"
    LOW_QUALITY_QUEUE_NAME = "low_quality_queue_name"
    ROI_BAD_CLASS_CONF_THRESHOLD = "roi_bad_class_confidence_threshold"
    REGIONS_TO_PROCESS = "regions_to_process"

def get_config_param(key, config_dict, default_value=None):
    if key in config_dict:
        return config_dict[key]

    return default_value


def get_updated_from_env_vars(old_config):
    # overwrite config values with the environment variables values (when those exists)

    config = {**old_config} #copy

    for k in config.keys():
        config[k] = os.environ.get(k) if k in os.environ else config[k]

    return config
