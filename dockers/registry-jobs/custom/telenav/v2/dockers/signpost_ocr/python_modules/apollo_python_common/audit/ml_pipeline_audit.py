from datetime import datetime
import logging
from google.protobuf.json_format import MessageToDict
import apollo_python_common.audit as audit
import apollo_python_common.ml_pipeline.config_api as config_api
from apollo_python_common.ml_pipeline.config_api import MQ_Param
from apollo_python_common.proto_api import MQ_Messsage_Type

LOG_BLACKLIST_PARAM_LIST = [MQ_Param.MQ_USERNAME, MQ_Param.MQ_PASSWORD]


class AuditParam:
    IMAGE_LOCATION = "image_location"
    PROTO_DATA = "proto_data"
    CLUSTER_LOCATION = "cluster_location"
    PROTO_DATA = "proto_data"
    TILE_TOP_LEFT = "tile_top_left"
    TILE_BOTTOM_RIGHT = "tile_bottom_right"
    IMAGE_SET_ROIS = "image_set_rois"
    TIMESTAMP = "timestamp"
    EXECUTION_DURATION = "execution_duration"
    RUN_CONFIGURATION = "run_configuration"
    CUSTOM_ARG = "custom_arg"
    PROCESSING_STAGE = "processing_stage"
    PROCESSING_STARTED = "started"
    PROCESSING_COMPLETED = "completed"


def __complete_doc_with_image_data(doc_base, image_proto):
    doc_base[AuditParam.IMAGE_LOCATION] = "{}, {}".format(image_proto.sensor_data.raw_position.latitude,
                                                          image_proto.sensor_data.raw_position.longitude)
    for key, value in MessageToDict(image_proto, preserving_proto_field_name=True).items():
        doc_base[f"{AuditParam.PROTO_DATA}_{key}"] = value
    return doc_base


def __complete_doc_with_cluster_data(doc_base, cluster_proto):
    doc_base[AuditParam.CLUSTER_LOCATION] = "{}, {}".format(cluster_proto.location.latitude,
                                           cluster_proto.location.longitude)
    for key, value in MessageToDict(cluster_proto, preserving_proto_field_name=True).items():
        doc_base[f"{AuditParam.PROTO_DATA}_{key}"] = value
    return doc_base


def __complete_doc_with_rois_data(doc_base, proto_data):
    roi_list = []
    for image in proto_data.image_set.images:
        for roi in image.rois:
            roi_list.append(roi.id)
    doc_base[f"{AuditParam.PROTO_DATA}_{AuditParam.IMAGE_SET_ROIS}"] = roi_list
    return doc_base


def __complete_doc_with_geotile_data(doc_base, geotile_proto):
    doc_base[f"{AuditParam.PROTO_DATA}_{AuditParam.TILE_TOP_LEFT}"] = "{}, {}".format(geotile_proto.top_left.latitude,
                                                                                      geotile_proto.top_left.longitude)
    doc_base[f"{AuditParam.PROTO_DATA}_{AuditParam.TILE_BOTTOM_RIGHT}"] = "{}, {}".format(geotile_proto.bottom_right.latitude,
                                                                                          geotile_proto.bottom_right.longitude)
    return doc_base


def __build_doc_list_for_proto(doc_base, proto_data):
    if type(proto_data).__name__ == str(MQ_Messsage_Type.IMAGE):
        doc_base = __complete_doc_with_image_data(doc_base, proto_data)
        return [doc_base]

    doc_list = []
    if type(proto_data).__name__ == str(MQ_Messsage_Type.GEO_TILE):
        doc_base = __complete_doc_with_geotile_data(doc_base, proto_data)
        if proto_data.clusters:
            # we have clusters -> stage completed
            for cluster in proto_data.clusters:
                doc_base = __complete_doc_with_cluster_data(doc_base, cluster)
                doc_list.append(doc_base.copy())
        else:
            # no clusters -> stage started, we should add roi id list
            doc_base = __complete_doc_with_rois_data(doc_base, proto_data)
            doc_list.append(doc_base)
    return doc_list


def __get_doc_list_from_data(proto_data, durations, run_configuration, custom_args=None):
    doc = {
        AuditParam.TIMESTAMP: datetime.now().isoformat(),
    }
    for (phase, duration) in durations:
        doc[f'{AuditParam.EXECUTION_DURATION}_{phase}'] = duration

    for key, value in run_configuration.items():
        if key in LOG_BLACKLIST_PARAM_LIST:
            continue
        doc[f"{AuditParam.RUN_CONFIGURATION}_{key}"] = value

    for key, value in custom_args.items():
        doc[f"{AuditParam.CUSTOM_ARG}_{key}"] = value

    return __build_doc_list_for_proto(doc, proto_data)


def __audit_one_message(proto_data, durations, run_configuration, custom_args=None):
    logger = logging.getLogger(__name__)
    es_enabled = config_api.get_config_param(MQ_Param.ELASTICSEARCH_ENABLED, run_configuration, True)

    if not es_enabled:
        return

    try:
        es_cnn = audit.connection()
        if es_cnn is None:
            logger.warning("Elasticsearch was not initialised. Impossible to store audit data.")
        else:
            doc_list = __get_doc_list_from_data(proto_data, durations, run_configuration, custom_args)
            for doc in doc_list:
                es_cnn.index(index=audit.audit_index_name(), doc_type='audit', body=doc)

    except Exception as er:
        logger.exception(er)


def one_message_was_processed(proto_data, durations, run_configuration, custom_args=None):
    if custom_args is None:
        custom_args = {}
    custom_args[AuditParam.PROCESSING_STAGE] = AuditParam.PROCESSING_COMPLETED
    __audit_one_message(proto_data, durations, run_configuration, custom_args)


def one_message_was_received(proto_data, run_configuration, custom_args=None):
    if custom_args is None:
        custom_args = {}
    custom_args[AuditParam.PROCESSING_STAGE] = AuditParam.PROCESSING_STARTED
    __audit_one_message(proto_data, [], run_configuration, custom_args)
