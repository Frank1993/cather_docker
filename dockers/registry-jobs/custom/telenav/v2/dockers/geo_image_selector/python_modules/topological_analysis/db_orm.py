# coding: utf-8
from sqlalchemy import BigInteger, Column, DateTime, Enum, Float, ForeignKey, ForeignKeyConstraint, Index, Integer, \
    Numeric, SmallInteger, String, text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class AccessToken(Base):
    __tablename__ = 'access_token'

    id = Column(String(191), primary_key=True)
    ttl = Column(Integer)
    created = Column(DateTime)
    userId = Column(Integer)


class Acl(Base):
    __tablename__ = 'acl'

    id = Column(Integer, primary_key=True)
    model = Column(String(512))
    property = Column(String(512))
    accessType = Column(String(512))
    permission = Column(String(512))
    principalType = Column(String(512))
    principalId = Column(String(512))


class AdasNode(Base):
    __tablename__ = 'adas_node'

    edge1_id = Column(BigInteger, primary_key=True, nullable=False)
    edge2_id = Column(BigInteger, primary_key=True, nullable=False)
    node_id = Column(BigInteger, primary_key=True, nullable=False)
    timestamp = Column(BigInteger)
    adas_c = Column(Float(asdecimal=True))
    adas_h = Column(Float(asdecimal=True))


class Area(Base):
    __tablename__ = 'area'

    area_id = Column(BigInteger, primary_key=True, unique=True)
    name = Column(String(255), unique=True)


class AreaBbox(Base):
    __tablename__ = 'area_bbox'

    area_bbox_id = Column(BigInteger, primary_key=True)
    min_lat = Column(Float(asdecimal=True))
    max_lat = Column(Float(asdecimal=True))
    min_lon = Column(Float(asdecimal=True))
    max_lon = Column(Float(asdecimal=True))
    area_id = Column(ForeignKey(u'area.area_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), index=True)

    area = relationship(u'Area')


class ComponentVersion(Base):
    __tablename__ = 'component_version'

    version_id = Column(Integer, primary_key=True)
    component_name = Column(Enum(u'lane', u'adas', u'road_sign'), nullable=False)
    timestamp = Column(BigInteger, nullable=False)
    component_version = Column(String(40), nullable=False)


class DrivableArea(Base):
    __tablename__ = 'drivable_area'
    __table_args__ = (
        ForeignKeyConstraint(['trip_id', 'trip_image_index'],
                             [u'trip_image_raw.trip_id', u'trip_image_raw.trip_image_index'], ondelete=u'CASCADE',
                             onupdate=u'CASCADE'),
        Index('trip_image_fk_idx', 'trip_id', 'trip_image_index')
    )

    drivable_adrea_id = Column(BigInteger, primary_key=True)
    trip_id = Column(BigInteger)
    trip_image_index = Column(BigInteger)
    point_index = Column(BigInteger)
    point_x = Column(Integer)
    point_y = Column(Integer)
    connected_component = Column(Integer)

    trip = relationship(u'TripImageRaw')


class Edge(Base):
    __tablename__ = 'edge'

    edge_id = Column(BigInteger, primary_key=True, nullable=False)
    node1_id = Column(BigInteger, primary_key=True, nullable=False)
    node2_id = Column(BigInteger, primary_key=True, nullable=False)
    timestamp = Column(BigInteger, nullable=False)


class EdgePoint(Base):
    __tablename__ = 'edge_point'

    edge_id = Column(BigInteger, primary_key=True, nullable=False)
    node1_id = Column(BigInteger, primary_key=True, nullable=False)
    node2_id = Column(BigInteger, primary_key=True, nullable=False)
    index = Column(BigInteger, primary_key=True, nullable=False)
    node_id = Column(BigInteger)
    latitude = Column(Float(asdecimal=True))
    longitude = Column(Float(asdecimal=True))
    elevation = Column(Float(asdecimal=True))
    adas_c = Column(Float(asdecimal=True))
    adas_h = Column(Float(asdecimal=True))
    adas_s = Column(Float(asdecimal=True))


class LaneAggregation(Base):
    __tablename__ = 'lane_aggregation'
    __table_args__ = (
        ForeignKeyConstraint(['edge_id', 'node1_id', 'node2_id'], [u'edge.edge_id', u'edge.node1_id', u'edge.node2_id'],
                             ondelete=u'CASCADE', onupdate=u'CASCADE'),
        Index('edge_fk_idx', 'edge_id', 'node1_id', 'node2_id')
    )

    lane_id = Column(BigInteger, primary_key=True)
    marker_left = Column(ForeignKey(u'marker_type.marker_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), nullable=False,
                         index=True)
    marker_right = Column(ForeignKey(u'marker_type.marker_id', ondelete=u'CASCADE', onupdate=u'CASCADE'),
                          nullable=False, index=True)
    confidence = Column(Float(asdecimal=True))
    edge_id = Column(BigInteger, nullable=False)
    node1_id = Column(BigInteger, nullable=False)
    node2_id = Column(BigInteger, nullable=False)

    edge = relationship(u'Edge')
    marker_type = relationship(u'MarkerType', primaryjoin='LaneAggregation.marker_left == MarkerType.marker_id')
    marker_type1 = relationship(u'MarkerType', primaryjoin='LaneAggregation.marker_right == MarkerType.marker_id')


class LaneAggregationAttribute(Base):
    __tablename__ = 'lane_aggregation_attributes'

    aggregation_attribute_id = Column(BigInteger, primary_key=True)
    attribute_id = Column(ForeignKey(u'lane_attributes.attribute_id', ondelete=u'CASCADE', onupdate=u'CASCADE'),
                          nullable=False, index=True)
    lane_id = Column(ForeignKey(u'lane_aggregation.lane_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), nullable=False,
                     index=True)

    attribute = relationship(u'LaneAttribute')
    lane = relationship(u'LaneAggregation')


class LaneAggregationDetectionLink(Base):
    __tablename__ = 'lane_aggregation_detection_link'

    link_id = Column(BigInteger, primary_key=True)
    aggregation_id = Column(ForeignKey(u'lane_aggregation.lane_id', ondelete=u'CASCADE', onupdate=u'CASCADE'),
                            nullable=False, index=True)
    detection_id = Column(
        ForeignKey(u'trip_image_lane_details.trip_image_lane_detection_id', ondelete=u'CASCADE', onupdate=u'CASCADE'),
        nullable=False, index=True)

    aggregation = relationship(u'LaneAggregation')
    detection = relationship(u'TripImageLaneDetail')


class LaneAttribute(Base):
    __tablename__ = 'lane_attributes'

    attribute_id = Column(Integer, primary_key=True)
    attribute = Column(
        Enum(u'none', u'left_arrow', u'right_arrow', u'forward_straight_arrow', u'forward_left_composed_arrow',
             u'forward_right_composed_arrow', u'backward_left_arrow', u'backward_right_arrow',
             u'backward_straight_arrow', u'backward_left_composed_arrow', u'backward_right_composed_arrow', u'bus',
             u'hov'), nullable=False, server_default=text("'none'"))


class LaneConnection(Base):
    __tablename__ = 'lane_connections'

    lane_connection_id = Column(BigInteger, primary_key=True)
    lane1_id = Column(ForeignKey(u'lane_aggregation.lane_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), nullable=False,
                      index=True)
    lane2_id = Column(ForeignKey(u'lane_aggregation.lane_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), nullable=False,
                      index=True)
    connection_node = Column(BigInteger, nullable=False)

    lane1 = relationship(u'LaneAggregation', primaryjoin='LaneConnection.lane1_id == LaneAggregation.lane_id')
    lane2 = relationship(u'LaneAggregation', primaryjoin='LaneConnection.lane2_id == LaneAggregation.lane_id')


class LaneDetectorCalibration(Base):
    __tablename__ = 'lane_detector_calibration'

    lane_detector_calib_id = Column(BigInteger, primary_key=True)
    trip_id = Column(ForeignKey(u'trip.trip_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), index=True)
    calibration_type = Column(Enum(u'CLASSIC', u'EGO_LANE', u'CLASSIC_FRAGMENTED', u'EGO_LANE_FRAGMENTED'))
    value = Column(Integer)
    yaw = Column(Float)
    pitch = Column(Float)
    roll = Column(Float)

    trip = relationship(u'Trip')


class MapVersion(Base):
    __tablename__ = 'map_version'

    map_version_id = Column(BigInteger, primary_key=True)
    timestamp = Column(BigInteger, nullable=False)


class MarkerType(Base):
    __tablename__ = 'marker_type'

    marker_id = Column(Integer, primary_key=True)
    marker_type = Column(
        Enum(u'SINGLE_SOLID', u'DOUBLE_SOLID', u'LONG_DASHED', u'SHORT_DASHED', u'SHADED_AREA', u'DASHED_BLOCKS',
             u'DOUBLE_LINE_DASHED_SOLID', u'DOUBLE_LINE_SOLID_DASHED', u'PHYSICAL_DIVIDER', u'DOUBLE_DASHED_LINES',
             u'UNKNOWN'))
    marker_color = Column(Enum(u'white', u'yellow'))


class NodeOsm(Base):
    __tablename__ = 'node_osm'

    node_id = Column(BigInteger, primary_key=True, unique=True)
    latitude = Column(Float(asdecimal=True))
    longitude = Column(Float(asdecimal=True))


class RoadSignType(Base):
    __tablename__ = 'road_sign_type'

    sign_id = Column(Integer, primary_key=True)
    sign_type = Column(String(50), unique=True)
    parent_sign_id = Column(ForeignKey(u'road_sign_type.sign_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), index=True)

    parent_sign = relationship(u'RoadSignType', remote_side=[sign_id])


class Role(Base):
    __tablename__ = 'role'

    id = Column(Integer, primary_key=True)
    name = Column(String(191), nullable=False)
    description = Column(String(191))
    created = Column(DateTime)
    modified = Column(DateTime)


class Rolemapping(Base):
    __tablename__ = 'rolemapping'

    id = Column(Integer, primary_key=True, nullable=False)
    principalType = Column(String(512))
    principalId = Column(Integer, primary_key=True, nullable=False, index=True)
    roleId = Column(ForeignKey(u'role.id', ondelete=u'CASCADE', onupdate=u'CASCADE'), index=True)

    role = relationship(u'Role')


class SignAggregation(Base):
    __tablename__ = 'sign_aggregation'

    aggregation_id = Column(BigInteger, primary_key=True)
    sign_type = Column(ForeignKey(u'road_sign_type.sign_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), nullable=False,
                       index=True)
    aggregation_confidence = Column(Float(asdecimal=True), nullable=False)
    latitude = Column(Float(asdecimal=True), nullable=False)
    longitude = Column(Float(asdecimal=True), nullable=False)
    way_id = Column(BigInteger, nullable=False)
    node1_id = Column(BigInteger, nullable=False)
    node2_id = Column(BigInteger, nullable=False)
    index = Column(BigInteger)
    matched_latitude = Column(Float(asdecimal=True), nullable=False)
    matched_longitude = Column(Float(asdecimal=True), nullable=False)
    facing = Column(Float(asdecimal=True), nullable=False)
    points = Column(String(45), nullable=False)
    validation_status = Column(
        Enum(u'true_positive', u'true_negative', u'false_positive', u'false_negative', u'unknown'),
        server_default=text("'unknown'"))

    road_sign_type = relationship(u'RoadSignType')


class SignAggregationDetectionLink(Base):
    __tablename__ = 'sign_aggregation_detection_link'

    link_id = Column(Integer, primary_key=True)
    aggregation_id = Column(BigInteger, nullable=False, index=True)
    detection_id = Column(
        ForeignKey(u'trip_image_sign_detection.trip_image_detection_id', ondelete=u'CASCADE', onupdate=u'CASCADE'),
        nullable=False, index=True)

    detection = relationship(u'TripImageSignDetection')


class Task(Base):
    __tablename__ = 'task'

    task_id = Column(BigInteger, primary_key=True)
    start_time = Column(BigInteger)
    end_time = Column(BigInteger)
    status = Column(Enum(u'NONE', u'EDITED', u'ALREADY DONE', u'NEED REVIEW', u'NOT IMAGE', u'INVALID'),
                    server_default=text("'NONE'"))
    description = Column(String(255))
    task_node_id = Column(ForeignKey(u'node_osm.node_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), index=True)
    asignee = Column(ForeignKey(u'user.user_id', ondelete=u'SET NULL', onupdate=u'CASCADE'), index=True)
    fc = Column(BigInteger)
    osv_coverage = Column(BigInteger)
    m_coverage = Column(BigInteger)
    area_id = Column(ForeignKey(u'area.area_id', ondelete=u'CASCADE'), index=True)
    type = Column(Enum(u'NONE', u'ROAD_SIGN', u'MAP_INTERSECTION'), server_default=text("'NONE'"))
    has_oneway = Column(Integer, server_default=text("'0'"))
    quad_key = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    road_sign_detection_id = Column(
        ForeignKey(u'sign_aggregation.aggregation_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), index=True)
    latitude = Column(Float(asdecimal=True), nullable=False)
    longitude = Column(Float(asdecimal=True), nullable=False)

    area = relationship(u'Area')
    user = relationship(u'User')
    road_sign_detection = relationship(u'SignAggregation')
    task_node = relationship(u'NodeOsm')


class TaskComment(Base):
    __tablename__ = 'task_comment'

    comment_id = Column(BigInteger, primary_key=True)
    user_id = Column(ForeignKey(u'user.user_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), index=True)
    task_id = Column(ForeignKey(u'task.task_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), index=True)
    comment = Column(String(255))
    created_at = Column(BigInteger, server_default=text("'0'"))

    task = relationship(u'Task')
    user = relationship(u'User')


class TaskHistory(Base):
    __tablename__ = 'task_history'

    history_id = Column(BigInteger, primary_key=True)
    user_id = Column(ForeignKey(u'user.user_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), nullable=False, index=True)
    task_id = Column(ForeignKey(u'task.task_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), nullable=False, index=True)
    field_changed = Column(String(255))
    old_value = Column(String(255))
    new_value = Column(String(255))
    created_at = Column(BigInteger, server_default=text("'0'"))

    task = relationship(u'Task')
    user = relationship(u'User')


class Trip(Base):
    __tablename__ = 'trip'

    trip_id = Column(BigInteger, primary_key=True)
    timestamp = Column(BigInteger, nullable=False)
    source_sequence_id = Column(BigInteger, nullable=False)
    user_id = Column(ForeignKey(u'user.user_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), index=True)
    obd2 = Column(Integer, nullable=False)
    status = Column(Enum(u'NOT_PROCESSED', u'PROCESSED', u'PROCESSING'))
    status_timestamp = Column(BigInteger)
    lat1 = Column(Float(asdecimal=True))
    lon1 = Column(Float(asdecimal=True))
    lat2 = Column(Float(asdecimal=True))
    lon2 = Column(Float(asdecimal=True))
    average_error = Column(Float(asdecimal=True))
    elevation_quality = Column(Float(asdecimal=True))

    user = relationship(u'User')


class TripDetectionRaw(Base):
    __tablename__ = 'trip_detection_raw'

    trip_id = Column(ForeignKey(u'trip.trip_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), primary_key=True,
                     nullable=False, index=True)
    trip_detection_index = Column(BigInteger, primary_key=True, nullable=False)
    timestamp = Column(BigInteger, nullable=False)
    type = Column(Integer, nullable=False)
    data = Column(Integer)
    latitude = Column(Float(asdecimal=True), nullable=False)
    longitude = Column(Float(asdecimal=True), nullable=False)
    elevation = Column(Float(asdecimal=True))
    h_accuracy = Column(Float(asdecimal=True))
    v_accuracy = Column(Float(asdecimal=True))
    heading = Column(Float(asdecimal=True))
    speed = Column(Float(asdecimal=True))

    trip = relationship(u'Trip')


class TripGpsPointRaw(Base):
    __tablename__ = 'trip_gps_point_raw'

    trip_id = Column(ForeignKey(u'trip.trip_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), primary_key=True,
                     nullable=False, index=True)
    trip_gps_index = Column(BigInteger, primary_key=True, nullable=False)
    timestamp = Column(BigInteger, nullable=False)
    latitude = Column(Float(asdecimal=True), nullable=False)
    longitude = Column(Float(asdecimal=True), nullable=False)
    elevation = Column(Float(asdecimal=True))
    h_accuracy = Column(Float(asdecimal=True))
    v_accuracy = Column(Float(asdecimal=True))
    heading = Column(Float(asdecimal=True))
    speed = Column(Float(asdecimal=True))

    trip = relationship(u'Trip')


class TripImageExif(Base):
    __tablename__ = 'trip_image_exif'
    __table_args__ = (
        ForeignKeyConstraint(['trip_id', 'trip_image_index'],
                             [u'trip_image_raw.trip_id', u'trip_image_raw.trip_image_index'], ondelete=u'CASCADE',
                             onupdate=u'CASCADE'),
        Index('trip_image_exif_trip_id_trip_photo_index', 'trip_id', 'trip_image_index')
    )

    trip_id = Column(BigInteger, primary_key=True, nullable=False)
    trip_image_index = Column(BigInteger, primary_key=True, nullable=False)
    key = Column(String(255), primary_key=True, nullable=False)
    value = Column(String(255), nullable=False)

    trip = relationship(u'TripImageRaw')


class TripImageLaneDetail(Base):
    __tablename__ = 'trip_image_lane_details'

    trip_image_lane_detection_id = Column(
        ForeignKey(u'trip_image_lane_detection.trip_image_lane_detection_id', ondelete=u'CASCADE', onupdate=u'CASCADE'),
        primary_key=True, nullable=False)
    lane_index = Column(SmallInteger, primary_key=True, nullable=False)
    left_divider_type = Column(ForeignKey(u'marker_type.marker_id', ondelete=u'CASCADE', onupdate=u'CASCADE'),
                               index=True)
    right_divider_type = Column(ForeignKey(u'marker_type.marker_id', ondelete=u'CASCADE', onupdate=u'CASCADE'),
                                index=True)
    left_divider_confidence = Column(Float(asdecimal=True))
    right_divider_confidence = Column(Float(asdecimal=True))
    direction = Column(Enum(u'forward', u'backward', u'unknown'), server_default=text("'unknown'"))
    bifurcation = Column(Enum(u'left down', u'left up', u'right down', u'right up', u'none'),
                         server_default=text("'none'"))
    lane_attributes = Column(ForeignKey(u'lane_attributes.attribute_id', ondelete=u'CASCADE', onupdate=u'CASCADE'),
                             index=True)
    current_lane = Column(Integer, server_default=text("'0'"))

    lane_attribute = relationship(u'LaneAttribute')
    marker_type = relationship(u'MarkerType',
                               primaryjoin='TripImageLaneDetail.left_divider_type == MarkerType.marker_id')
    marker_type1 = relationship(u'MarkerType',
                                primaryjoin='TripImageLaneDetail.right_divider_type == MarkerType.marker_id')
    trip_image_lane_detection = relationship(u'TripImageLaneDetection')


class TripImageLaneDetection(Base):
    __tablename__ = 'trip_image_lane_detection'
    __table_args__ = (
        ForeignKeyConstraint(['trip_id', 'trip_image_index'],
                             [u'trip_image_raw.trip_id', u'trip_image_raw.trip_image_index'], ondelete=u'CASCADE',
                             onupdate=u'CASCADE'),
        Index('lane_detection_trip_fk_idx', 'trip_id', 'trip_image_index')
    )

    trip_image_lane_detection_id = Column(BigInteger, primary_key=True, nullable=False)
    timestamp = Column(BigInteger, primary_key=True, nullable=False)
    trip_id = Column(BigInteger, nullable=False)
    trip_image_index = Column(BigInteger, nullable=False, index=True)
    lane_count = Column(Integer)
    detection_mode = Column(Enum(u'auto', u'manual', u'auto_manual_validation', u'unknown'),
                            server_default=text("'unknown'"))
    processing_status = Column(Enum(u'pending', u'processed', u'in set'), server_default=text("'pending'"))
    validation_status = Column(
        Enum(u'true_positive', u'true_negative', u'false_positive', u'false_negative', u'unknown'),
        server_default=text("'unknown'"))
    user_id = Column(ForeignKey(u'user.user_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), nullable=False, index=True)
    calibration_params = Column(ForeignKey(u'lane_detector_calibration.lane_detector_calib_id'), index=True)
    lane_version = Column(ForeignKey(u'component_version.version_id'), index=True)

    lane_detector_calibration = relationship(u'LaneDetectorCalibration')
    component_version = relationship(u'ComponentVersion')
    trip = relationship(u'TripImageRaw')
    user = relationship(u'User')


class TripImageRaw(Base):
    __tablename__ = 'trip_image_raw'

    trip_id = Column(ForeignKey(u'trip.trip_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), primary_key=True,
                     nullable=False, index=True)
    trip_image_index = Column(BigInteger, primary_key=True, nullable=False)
    timestamp = Column(BigInteger, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)

    trip = relationship(u'Trip')


class TripImage(TripImageRaw):
    __tablename__ = 'trip_image'
    __table_args__ = (
        ForeignKeyConstraint(['trip_id', 'trip_image_index'],
                             [u'trip_image_raw.trip_id', u'trip_image_raw.trip_image_index'], ondelete=u'CASCADE',
                             onupdate=u'CASCADE'),
        Index('trip_image_trip_id_trip_photo_index', 'trip_id', 'trip_image_index')
    )

    trip_id = Column(BigInteger, primary_key=True, nullable=False)
    trip_image_index = Column(BigInteger, primary_key=True, nullable=False)
    trip_image_set_id = Column(
        ForeignKey(u'trip_image_set.trip_image_set_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), index=True)
    trip_point_index = Column(ForeignKey(u'trip_point.trip_point_index', ondelete=u'CASCADE', onupdate=u'CASCADE'),
                              index=True)

    trip_image_set = relationship(u'TripImageSet')
    trip_point = relationship(u'TripPoint')


class TripImageRoi(Base):
    __tablename__ = 'trip_image_roi'
    __table_args__ = (
        ForeignKeyConstraint(['trip_id', 'trip_image_index'], [u'trip_image.trip_id', u'trip_image.trip_image_index'],
                             ondelete=u'CASCADE', onupdate=u'CASCADE'),
        Index('trip_image_fk_idx', 'trip_id', 'trip_image_index')
    )

    trip_image_roi_id = Column(BigInteger, primary_key=True, nullable=False)
    user_id = Column(ForeignKey(u'user.user_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), primary_key=True,
                     nullable=False, index=True)
    trip_id = Column(ForeignKey(u'trip.trip_id'), nullable=False)
    trip_image_index = Column(BigInteger, nullable=False)
    timestamp = Column(BigInteger, nullable=False)
    x1 = Column(Float(5, True), nullable=False)
    y1 = Column(Float(5, True), nullable=False)
    width = Column(Float(5, True), nullable=False)
    height = Column(Float(5, True), nullable=False)
    detection_mode = Column(Enum(u'auto', u'manual', u'auto_manual_validation', u'unknown'),
                            server_default=text("'unknown'"))
    processing_status = Column(Enum(u'pending', u'processed', u'in set'), server_default=text("'pending'"))
    validation_status = Column(
        Enum(u'true_positive', u'true_negative', u'false_positive', u'false_negative', u'unknown'),
        server_default=text("'unknown'"))
    sign_aggregation_id = Column(
        ForeignKey(u'sign_aggregation.aggregation_id', ondelete=u'SET NULL', onupdate=u'CASCADE'), index=True)
    sign_detection_version = Column(ForeignKey(u'component_version.version_id'), index=True)
    modify_timestamp = Column(DateTime, nullable=False,
                              server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))

    sign_aggregation = relationship(u'SignAggregation')
    component_version = relationship(u'ComponentVersion')
    trip = relationship(u'TripImage')
    trip1 = relationship(u'Trip')
    user = relationship(u'User')


class TripImageSet(Base):
    __tablename__ = 'trip_image_set'

    trip_image_set_id = Column(BigInteger, primary_key=True)
    set_type = Column(Enum(u'none', u'training', u'metrics'), nullable=False)
    timestamp = Column(BigInteger)
    type_of_detection = Column(ForeignKey(u'road_sign_type.sign_id', ondelete=u'CASCADE', onupdate=u'CASCADE'),
                               index=True)

    road_sign_type = relationship(u'RoadSignType')


class TripImageSignDetection(Base):
    __tablename__ = 'trip_image_sign_detection'

    trip_image_detection_id = Column(BigInteger, primary_key=True)
    trip_image_roi_id = Column(
        ForeignKey(u'trip_image_roi.trip_image_roi_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), nullable=False,
        index=True)
    type_of_detection = Column(ForeignKey(u'road_sign_type.sign_id', ondelete=u'CASCADE', onupdate=u'CASCADE'),
                               nullable=False, index=True)
    confidence_level = Column(Float)
    settings_type = Column(
        Enum(u'SPEED_LIMIT_US_SL', u'SPEED_LIMIT_EU_SL', u'LICENSE_PLATE_CV', u'LICENSE_PLATE_ALPR', u'FACES_HAAR_CV',
             u'FACES_LBP_CV', u'TURN_RESTRICTIONS_US_SL', u'REGULATORY_CANADA_SL', u'RED_CIRCLES_GTSRB_SVM',
             u'RED_CIRCLES_RUS_SVM', u'STOP_SIGNS_SL', u'REGULATORY_SL', u'RUSSIAN_CAFFE', u'RED_CIRCLES_RUS_MSER_SVM',
             u'RED_CIRCLES_MSER_RTREES', u'CLRSEP_RTREES', u'HIGHWAY_PANELS'))
    filter_type = Column(Enum(u'red', u'blue', u'green', u'white', u'blue_white', u'orange', u'radial_symetry',
                              u'reversed_radial_symetry', u'MSER', u'undefined'))
    classifier_type = Column(
        Enum(u'undefined', u'give_way', u'highway_sign', u'regulatory_direction', u'canada_regulatory_direction',
             u'speed_limit_eu', u'speed_limit_construction_us', u'speed_limit_us', u'stop_sign', u'turn_restriction',
             u'SVMC_eu', u'SVMC_us', u'caffe_eu'))
    classification_validation = Column(Enum(u'unknown', u'positive', u'negative'))
    active = Column(Integer, nullable=False, server_default=text("'1'"))
    distance = Column(Float)
    angle_of_roi = Column(Float)
    angle_from_center = Column(Float)
    detected_lat = Column(Float)
    detected_lon = Column(Float)
    parent_sign_id = Column(
        ForeignKey(u'trip_image_sign_detection.trip_image_detection_id', ondelete=u'CASCADE', onupdate=u'CASCADE'),
        index=True)
    text = Column(String(100))
    way_snip = Column(String(45))
    osm_relation_sniped = Column(String(45))

    parent_sign = relationship(u'TripImageSignDetection', remote_side=[trip_image_detection_id])
    trip_image_roi = relationship(u'TripImageRoi')
    road_sign_type = relationship(u'RoadSignType')


class TripPoint(Base):
    __tablename__ = 'trip_point'

    trip_point_index = Column(BigInteger, primary_key=True, nullable=False, unique=True)
    trip_id = Column(ForeignKey(u'trip.trip_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), primary_key=True,
                     nullable=False, index=True)
    timestamp = Column(BigInteger, nullable=False)
    latitude = Column(Float(asdecimal=True), nullable=False)
    longitude = Column(Float(asdecimal=True), nullable=False)
    elevation = Column(Float(asdecimal=True))
    match_latitude = Column(Float(asdecimal=True))
    match_longitude = Column(Float(asdecimal=True))
    match_edge_id = Column(BigInteger)
    match_node1_id = Column(BigInteger)
    match_node2_id = Column(BigInteger)
    heading = Column(Integer, nullable=False)
    pitch = Column(Float(asdecimal=True), nullable=False)
    yaw = Column(Float(asdecimal=True), nullable=False)
    roll = Column(Float(asdecimal=True), nullable=False)
    point_location = Column(Float(asdecimal=True), nullable=False)
    radius = Column(Float(asdecimal=True), nullable=False)
    adas_version = Column(ForeignKey(u'component_version.version_id', ondelete=u'CASCADE', onupdate=u'CASCADE'),
                          index=True)

    component_version = relationship(u'ComponentVersion')
    trip = relationship(u'Trip')


class TripSensorRaw(Base):
    __tablename__ = 'trip_sensor_raw'

    trip_id = Column(ForeignKey(u'trip.trip_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), primary_key=True,
                     nullable=False, index=True)
    trip_sensor_index = Column(BigInteger, primary_key=True, nullable=False)
    timestamp = Column(BigInteger, nullable=False)
    yaw = Column(Float(asdecimal=True))
    pitch = Column(Float(asdecimal=True))
    roll = Column(Float(asdecimal=True))
    acceleration_x = Column(Float(asdecimal=True))
    acceleration_y = Column(Float(asdecimal=True))
    acceleration_z = Column(Float(asdecimal=True))
    gravity_x = Column(Float(asdecimal=True))
    gravity_y = Column(Float(asdecimal=True))
    gravity_z = Column(Float(asdecimal=True))
    air_pressure = Column(Float(asdecimal=True))
    compass = Column(Float(asdecimal=True))
    obs_speed = Column(Float(asdecimal=True))
    oxc_steering_wheel_angle = Column(Float(asdecimal=True))

    trip = relationship(u'Trip')


class User(Base):
    __tablename__ = 'user'

    user_id = Column(BigInteger, primary_key=True)
    type = Column(Enum(u'auto', u'human'))
    external_id = Column(BigInteger, server_default=text("'0'"))
    username = Column(String(100))
    trust_level = Column(Numeric(5, 4))
    realm = Column(String(512))
    password = Column(String(512), nullable=False)
    email = Column(String(512), nullable=False)
    emailVerified = Column(Integer)
    verificationToken = Column(String(512))


class UserAcces(User):
    __tablename__ = 'user_access'

    user_id = Column(ForeignKey(u'user.user_id', ondelete=u'CASCADE', onupdate=u'CASCADE'), primary_key=True)
    last_request_timestamp = Column(BigInteger, nullable=False)
    request_count = Column(BigInteger, nullable=False)
