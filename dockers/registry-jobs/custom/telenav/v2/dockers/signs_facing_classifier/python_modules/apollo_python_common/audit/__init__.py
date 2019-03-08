from elasticsearch import Elasticsearch
from apollo_python_common.ml_pipeline.config_api import get_config_param, MQ_Param

_connection = None
_audit_index_name = None


def init(conf=None):
    if conf is None:
        conf = {}

    es_enabled = get_config_param(MQ_Param.ELASTICSEARCH_ENABLED, conf, True)

    if not es_enabled:
        return

    global _connection, _audit_index_name
    # Init connection
    if _connection is None:
        es_hostname = get_config_param(MQ_Param.ELASTICSEARCH_HOST, conf, "localhost")
        if es_hostname is not None:
            _connection = Elasticsearch([es_hostname])

    if _audit_index_name is None:
        _audit_index_name = get_config_param(MQ_Param.ELASTICSEARCH_AUDIT_INDEX_NAME, conf, "audit_ml_pipeline")

    ensure_index_exists()


def ensure_index_exists():
    if not _connection.indices.exists(index=_audit_index_name):
        audit_mapping = {"mappings": {"audit": {"properties": {"image_location": {"type": "geo_point"}}}}}
        _connection.indices.create(index=_audit_index_name, body=audit_mapping)


def connection():
    """Return the Elasticsearch connection to the host name given by the environment
    variable elasticsearch_host, creating it if necessary.

    """
    if _connection is None:
        init()
    return _connection


def audit_index_name():
    if _audit_index_name is None:
        init()
    return _audit_index_name
