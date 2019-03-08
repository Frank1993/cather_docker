from trip_downloader.constants import *
import logging


def get_in_clause(value, negated=False):
    clause = ' in '
    if negated:
        clause = ' not' + clause
    clause += value
    return clause


def add_condition(where_clause, column_name, operator, value, negated=False):
    if value == ['all']:
        return where_clause

    where_clause += ' and ' + column_name

    if len(value) > 1:
        added_clause = get_in_clause(str(tuple(value)), negated)
    else:
        added_clause = ' ' + operator + ' \'{}\''.format(value[0])

    where_clause += added_clause
    return where_clause


def add_sign_type_condition(where_clause, config_data):
    signs = config_data.signs
    if signs == ['all']:
        regions = config_data.regions

        if len(regions) == 1:
            where_clause += " and sign.region = '{}'".format(regions[0])
        elif len(regions) > 1:
            sign_region_clause = get_in_clause(str(tuple(regions)), False)
            where_clause += " and (sign.region {})".format(sign_region_clause)

        if not config_data.sign_components:
            where_clause += ' and detection.parent_id IS NULL '
        return where_clause

    where_clause += ' and (sign.internal_name '

    if len(signs) > 1:
        sign_type_clause = get_in_clause(str(tuple(signs)), False)
    else:
        sign_type_clause = ' = \'{}\''.format(signs[0])
    where_clause += sign_type_clause

    if config_data.sign_components:
        where_clause += ' or parent_sign.internal_name ' + sign_type_clause

    where_clause += ')'
    return where_clause


def get_mode_list(config_data):
    mode_list = list()
    if config_data.automatic:
        mode_list.append(AUTOMATIC)
    if config_data.manual:
        mode_list.append(MANUAL)
    return mode_list


def get_validation_list(config_data):
    validation_list = list()
    if config_data.confirmed:
        validation_list.append(CONFIRMED)
    if config_data.removed:
        validation_list.append(REMOVED)
    if config_data.to_be_checked:
        validation_list.append(TO_BE_CHECKED)
    return validation_list


def get_where_data(config_data):
    where_clause = 'where 1 = 1'
    where_clause = add_condition(where_clause, "detection.sequence_id", "=", config_data.trip_ids_included)
    where_clause = add_condition(where_clause, "detection.sequence_id", "!=", config_data.trip_ids_excluded, True)

    where_clause = add_sign_type_condition(where_clause, config_data)

    mode_list = get_mode_list(config_data)
    where_clause = add_condition(where_clause, "detection.mode", "=", mode_list)

    validation_list = get_validation_list(config_data)
    where_clause = add_condition(where_clause, "detection.validation_status", "=", validation_list)

    if config_data.min_signs_size > 0:
        min_signs_size = [config_data.min_signs_size]
        where_clause = add_condition(where_clause, "detection.width * photo.width", ">", min_signs_size)
        where_clause = add_condition(where_clause, "detection.height * photo.height", ">", min_signs_size)

    if config_data.telenav_user:
        where_clause = add_condition(where_clause, "coalesce(TLU.last_user, 'TELENAV')",  "<>", ['OTHER'])

    if config_data.gte_timestamp != BOT:
        where_clause = add_condition(where_clause, "sequence.creation_timestamp", ">=", [config_data.gte_timestamp])
    if config_data.lte_timestamp != NOW:
        where_clause = add_condition(where_clause, "sequence.creation_timestamp", "<=", [config_data.lte_timestamp])

    return where_clause


def get_query(config_data):
    logger = logging.getLogger(__name__)
    base_query = """
    WITH T_max_contr_id AS
        (
        SELECT  
            d.sequence_id, d.sequence_index, max(c.id) AS max_contr_id
        FROM contribution c
        JOIN detection d ON d.id = c.detection_id
        JOIN edit e ON c.id=e.contribution_id
        WHERE e.type != 'EDIT_STATUS_CHANGE'
        GROUP BY d.sequence_id, d.sequence_index),
        T_last_user AS (
            SELECT 
            -- Selecting those pictures where last user for a picture is telenav
            TMC.sequence_id, TMC.sequence_index,
            case when auth.username like '%telenav' or 
                        auth.username in ('istvan_bardos', 'anca_mihaela', 'judith92', 'tudor_adam',
                                            'alexandru_bura','gavan_ionut', 'carmen_nicula','catalin_groza', 
                                            'Steve', 'mihaela_moldovan', 'baditaflorin','DonPanda','11010101',
                                            'angela_ssafira', 'mihaiserban') then 'TELENAV' else 'OTHER'
            end as last_user																	
            FROM contribution C
            JOIN T_max_contr_id TMC on TMC.max_contr_id = C.id
            JOIN author auth ON auth.id = C.author_id
        )
    SELECT 
        detection.sequence_id, detection.sequence_index, 
        coalesce(st_x(photo.point), 0) AS longitude, coalesce(st_y(photo.point), 0) AS latitude, 
        coalesce(photo.width, 0) AS photo_width, coalesce(photo.height, 0) AS photo_height, 
        coalesce(photo.heading, 0) as heading, 
        coalesce(photo.osc_id, -1) as osc_id, 
        coalesce(photo.gps_accuracy, 0) as gps_acc,
        detection.id AS detection_id, detection.x, detection.y, detection.width AS detection_width, 
        detection.height AS detection_height, detection.confidence_level, 
        detection.validation_status, detection.mode, 
        coalesce(st_x(detection.point), 0) AS detection_longitude, 
        coalesce(st_y(detection.point), 0) AS detection_latitude,
        coalesce(detection.facing, 0) AS facing, coalesce(detection.distance, 0) AS distance,
        coalesce(detection.orientation, 0) AS angle_of_roi,
        sign.internal_name, sign.region, detection.parent_id
    FROM detection
    LEFT OUTER JOIN photo ON photo.sequence_id = detection.sequence_id and 
                              photo.sequence_index = detection.sequence_index 
    LEFT OUTER JOIN T_last_user TLU ON photo.sequence_id = TLU.sequence_id and
                                        photo.sequence_index = TLU.sequence_index 
    JOIN sign ON detection.sign_id = sign.id
    LEFT OUTER JOIN sequence ON photo.sequence_id = sequence.id
    LEFT OUTER JOIN detection parent_detection ON detection.parent_id = parent_detection.id
    LEFT OUTER JOIN sign parent_sign ON parent_detection.sign_id = parent_sign.id
    {} 
    ORDER BY detection.sequence_id, detection.sequence_index"""

    where_clause = get_where_data(config_data)
    sql_query = base_query.format(where_clause)
    logger.info(sql_query)
    return sql_query
