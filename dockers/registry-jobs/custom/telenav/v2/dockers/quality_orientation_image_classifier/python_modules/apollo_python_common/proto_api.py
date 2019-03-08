import os
from collections import defaultdict
from google.protobuf import text_format

import orbb_camera_specification_pb2
import orbb_tracking_pb2
import orbb_metadata_pb2
import clustering_pb2
from orbb_definitions_pb2 import _MARK as ROI_MARK, _SIGNELEMENT as SIGN_ELEMENT, _VALIDATION as ROI_VALIDATION
import apollo_python_common.io_utils as io_utils
from apollo_python_common.rectangle import Rectangle
import apollo_python_common.proto_utils.duplicate_rois_removal as duplicate_rois_removal


class MQ_Messsage_Type:
    IMAGE = "Image"
    GEO_TILE = "GeoTile"


def get_new_imageset_proto(imageset_name="imageset"):
    metadata = orbb_metadata_pb2.ImageSet()
    metadata.name = imageset_name
    return metadata


def get_new_image_proto(trip_id, image_index, image_path, region, latitude, longitude, serialized=False):
    image = orbb_metadata_pb2.Image()
    image.metadata.trip_id = trip_id
    image.metadata.image_index = image_index
    image.metadata.image_path = image_path
    image.metadata.region = region
    image.metadata.id = "0"
    image.sensor_data.raw_position.latitude = latitude
    image.sensor_data.raw_position.longitude = longitude
    image.sensor_data.heading = 0.0
    image.sensor_data.timestamp = 0
    image.sensor_data.speed_kmh = 0.0
    if serialized:
        return image.SerializeToString()
    else:
        return image


def get_new_processed_sign_localization_proto(latitude, longitude, distance, angle_from_center, facing, angle_of_roi):
    local = orbb_metadata_pb2.ProcessedSignLocalization()
    local.position.latitude = latitude
    local.position.longitude = longitude
    local.distance = distance
    local.angle_from_center = angle_from_center
    local.facing = facing
    local.angle_of_roi = angle_of_roi

    return local


def read_imageset_file(file_name):
    metadata = orbb_metadata_pb2.ImageSet()
    with open(file_name, 'rb') as f:
        metadata.ParseFromString(f.read())
    return metadata


def read_image_proto(serialized_image_proto):
    image = orbb_metadata_pb2.Image()
    image.ParseFromString(serialized_image_proto)
    return image


def merge_imageset_files(meta_file_1, meta_file_2):
    roi_metadata1 = read_imageset_file(meta_file_1)
    roi_dict1 = create_images_dictionary(roi_metadata1)
    roi_metadata2 = read_imageset_file(meta_file_2)
    roi_dict2 = create_images_dictionary(roi_metadata2)
    roi_dict_merged = merge_image_dictionaries(roi_dict1, roi_dict2)
    merged_meta = create_imageset_from_dict(roi_dict_merged)
    return merged_meta


def add_mandatory_sensor_data_to_proto(image_proto, longitude, latitude, heading, width, height, timestamp, speed_kmh, device_type):
    image_proto.sensor_data.raw_position.longitude = longitude
    image_proto.sensor_data.raw_position.latitude = latitude
    image_proto.sensor_data.heading = heading
    image_proto.sensor_data.img_res.width = width
    image_proto.sensor_data.img_res.height = height
    image_proto.sensor_data.timestamp = timestamp
    image_proto.sensor_data.speed_kmh = speed_kmh
    image_proto.sensor_data.device_type = device_type
    return image_proto


def add_mandatory_metadata_to_proto(image_proto, image_path, trip_id, img_indx, region):
    image_proto.metadata.image_path = image_path
    image_proto.metadata.trip_id = trip_id
    image_proto.metadata.image_index = img_indx
    image_proto.metadata.region = region
    image_proto.metadata.id = "0"
    return image_proto


def add_image_size(image_proto, img_shape):
    '''
    Adds image size info into sensor data
    :param image_proto: the Image proto instance
    :param img_shape: the shape of the image to be stored as metadata into image's protobuf
    :return: None
    '''
    height, width = img_shape[:2]
    img_res = image_proto.sensor_data.img_res
    if img_res.width == 0 or img_res.height == 0:
        img_res.width = width
        img_res.height = height


def serialize_proto_instance(proto_instance, output_path, file_name="rois"):
    io_utils.create_folder(output_path)
    with open(os.path.join(output_path, "{}.bin".format(file_name)), "wb") as f:
        f.write(proto_instance.SerializeToString())

    with open(os.path.join(output_path, "{}.txt".format(file_name)), "w") as f:
        f.write(str(proto_instance))


def append_new_image_to_imageset(imageset, img_filename, img_rois):
    image = imageset.images.add()
    image.metadata.image_path = os.path.basename(img_filename)
    image.metadata.trip_id = ""
    image.metadata.image_index = 0
    image.metadata.region = ""
    image.metadata.id = "0"
    image.sensor_data.raw_position.latitude = 0
    image.sensor_data.raw_position.longitude = 0
    image.sensor_data.heading = 0.0
    image.sensor_data.timestamp = 0
    image.sensor_data.speed_kmh = 0.0
    for current_roi in img_rois:
        append_roi(image, current_roi)


def check_imageset(rois_file):
    roi_metadata = read_imageset_file(rois_file)
    roi_dict = create_images_dictionary(roi_metadata)
    all_types_counts = dict()
    for file_name, rois in roi_dict.items():
        for roi in rois:
            roi_class = ROI_MARK.values_by_number[roi.type].name
            count = all_types_counts.get(roi_class, 0)
            all_types_counts[roi_class] = count + 1
            count = all_types_counts.get('total', 0)
            all_types_counts['total'] = count + 1
    return all_types_counts


def create_images_dictionary(metadata, check_validation=True):
    dictionary = dict()
    for element in metadata.images:
        file_name = os.path.basename(element.metadata.image_path)
        dictionary[file_name] = list()
        for roi in element.rois:
            if check_validation:
                if roi.validation == 0:
                    dictionary[file_name].append(roi)
            else:
                dictionary[file_name].append(roi)
    return dictionary


def create_imageset_from_dict(dict_meta):
    metadata = get_new_imageset_proto()
    for filename, rois in dict_meta.items():
        append_new_image_to_imageset(metadata, filename, rois)
    return metadata


def merge_image_dictionaries(dict1, dict2):
    for filename, rois in dict2.items():
        if len(rois) > 0:
            dict1[filename].extend(rois)
    return dict1


def get_filtered_imageset_dict(rois_file, selected_classes):
    image_set = read_imageset_file(rois_file)
    image_set = remove_duplicate_rois(image_set)
    roi_dict = create_images_dictionary(image_set)
    result_dict = defaultdict(list)
    for file_base_name, rois in roi_dict.items():
        remaining_rois = list()
        for roi in rois:
            if get_roi_type_name(roi.type) in selected_classes:
                remaining_rois.append(roi)
        if len(remaining_rois) > 0:
            result_dict[file_base_name] = remaining_rois
    return result_dict


def get_class_names_from_images_dictionary(rois_dict, translate_to_sign_components=False):
    all_classes = set()
    for file_name, rois in rois_dict.items():
        for roi in rois:
            class_name = get_component_type_name(roi.type) if translate_to_sign_components else get_roi_type_name(roi.type)
            all_classes.add(class_name)
    return sorted(list(all_classes))


def append_roi(image, current_roi):
    roi = image.rois.add()
    roi.type = current_roi.type
    roi.rect.CopyFrom(current_roi.rect)
    roi.manual = current_roi.manual
    roi.algorithm = current_roi.algorithm
    roi.algorithm_version = current_roi.algorithm_version
    for detection in current_roi.detections:
        new_detection = roi.detections.add()
        new_detection.type = detection.type
        new_detection.confidence = detection.confidence
    for component in current_roi.components:
        new_component = roi.components.add()
        new_component.type = component.type
        new_component.box.CopyFrom(component.box)
        new_component.value = component.value
    roi.validation = current_roi.validation


def append_rois_to_existing_image(imageset, img_filename, img_rois):
    image_idx = [i for i, v in enumerate(imageset.images) if v.metadata.image_path == os.path.basename(img_filename)]
    for current_roi in img_rois:
        append_roi(imageset.images[image_idx[0]], current_roi)


def get_confident_rois(rois_dict, thresholds_per_class):
    selected_dict = dict()
    for file_name, rois in rois_dict.items():
        rois_list = list()
        selected_dict[file_name] = rois_list
        for roi in rois:
            if roi.detections[0].confidence > thresholds_per_class[get_roi_type_name(roi.type)]:
                rois_list.append(roi)
    return selected_dict


def get_roi_type_name(type_value):
    return ROI_MARK.values_by_number[type_value].name


def get_roi_validation_value(validation_name):
    return ROI_VALIDATION.values_by_name[validation_name].number


def get_roi_type_value(type_name):
    if type_name in ROI_MARK.values_by_name.keys():
        return ROI_MARK.values_by_name[type_name].number
    else:
        return ROI_MARK.values_by_name["INVALID"].number


def remove_duplicate_rois(imageset_proto):
    return duplicate_rois_removal.remove_duplicate_rois(imageset_proto)


def filter_rois_by_classes(selected_classes, all_rois_dict):
    all_selected_rois = dict()
    for file_name, rois in all_rois_dict.items():
        selected_rois = list()
        for roi in rois:
            label, rect = get_rect_from_roi(roi)
            if label in selected_classes:
                selected_rois.append(roi)
        all_selected_rois[file_name] = selected_rois
    return all_selected_rois


def get_rect_from_roi(roi):
    label = ROI_MARK.values_by_number[roi.type].name
    xmin = roi.rect.tl.col
    ymin = roi.rect.tl.row
    xmax = roi.rect.br.col
    ymax = roi.rect.br.row
    bounding_box = Rectangle(xmin, ymin, xmax, ymax)
    return label, bounding_box


def get_component_type_name(type_value):
    return SIGN_ELEMENT.values_by_number[type_value].name


def get_component_type_value(type_name):
    return SIGN_ELEMENT.values_by_name[type_name].number


def add_missing_images(base_image_set_path, image_set_to_add_path):
    base_image_set = read_imageset_file(base_image_set_path)
    image_set_to_add = read_imageset_file(image_set_to_add_path)
    image_list = set([(image.metadata.trip_id, image.metadata.image_index) for image in base_image_set.images])
    for image in image_set_to_add.images:
        if (image.metadata.trip_id, image.metadata.image_index) not in image_list:
            base_image_set.images.extend([image])

    return base_image_set


def get_new_cluster_proto():
    cluster_proto = clustering_pb2.Clusters()
    return cluster_proto


def read_clusters_file(clusters_file_name):
    clusters = clustering_pb2.Clusters()
    with open(clusters_file_name, 'rb') as f:
        clusters.ParseFromString(f.read())
    return clusters


def get_new_geotile_proto():
    geotile_proto = clustering_pb2.GeoTile()
    return geotile_proto


def read_geotile_proto(serialized_geotile_proto):
    geotile = clustering_pb2.GeoTile()
    geotile.ParseFromString(serialized_geotile_proto)
    return geotile


def read_geotile_file(geotile_file_name):
    geotile = clustering_pb2.GeoTile()
    with open(geotile_file_name, 'rb') as f:
        geotile.ParseFromString(f.read())
    return geotile


def add_vanishing_point(image_proto, img, vp_detector, force_recalculate=False):
    '''
    Adds vanishing point (VP) info into images's features
    :param image_proto: the Image proto instance
    :param img: the image in BGR format
    :param vp_detector: an instance of VanishingPointDetector
    :param force_recalculate: whether VP should be recalculated if already exists
    :return: None
    '''
    if force_recalculate or image_proto.features.vanishing_point.confidence == 0:
        # confidence is 0 when VP was not calculated before
        detected_vp, confidence = vp_detector.get_vanishing_point(img)
        if detected_vp is not None:
            image_proto.features.vanishing_point.confidence = confidence
            image_proto.features.vanishing_point.vp.row = max(0, detected_vp.y)
            image_proto.features.vanishing_point.vp.col = max(0, detected_vp.x)


def set_vanishing_point(image_proto, vanishing_point, confidence, force_recalculate=False):
    if force_recalculate or image_proto.features.vanishing_point.confidence == 0:
        # confidence is 0 when VP was not calculated before
        if vanishing_point is not None:
            image_proto.features.vanishing_point.confidence = confidence
            image_proto.features.vanishing_point.vp.row = max(0, vanishing_point.y)
            image_proto.features.vanishing_point.vp.col = max(0, vanishing_point.x)


def image_has_matched_position(image_proto):
    return image_proto.HasField("match_data") and image_proto.match_data.HasField("matched_position") and \
           image_proto.match_data.HasField("matched_heading")


def image_has_valid_resolution(image_proto):
    return image_proto.sensor_data.HasField("img_res") and image_proto.sensor_data.img_res.width != 0 \
           and image_proto.sensor_data.img_res.height != 0


def empty_roi_list(image_proto):
    return len(image_proto.rois) == 0


def add_sign_position_to_roi(sign_position, roi):
    roi.local.distance = sign_position.distance
    roi.local.angle_from_center = sign_position.angle_from_center
    roi.local.position.latitude = sign_position.latitude
    roi.local.position.longitude = sign_position.longitude
    roi.local.facing = sign_position.facing


def valid_roi(roi):
    width = roi.rect.br.col - roi.rect.tl.col
    height = roi.rect.br.row - roi.rect.tl.row
    return width != 0 and height != 0


def read_phone_lenses(proto_file):
    phone_lenses_list = orbb_camera_specification_pb2.PhoneLenses()
    with open(proto_file, 'rb') as f:
        text_format.Parse(f.read().decode('UTF-8'), phone_lenses_list, allow_unknown_extension=True)
    return phone_lenses_list


def create_phone_lenses_dict(phone_lenses_protobuf):
    phone_lenses_dict = {}
    for phone_lense in phone_lenses_protobuf.phone_lenses:
        phone_lenses_dict[phone_lense.device_name] = phone_lense
    return phone_lenses_dict


def read_sign_dimensions(proto_file):
    sign_dimensions_list = orbb_tracking_pb2.TrackingConfigMeta()
    with open(proto_file, 'rb') as f:
        text_format.Parse(f.read().decode('UTF-8'), sign_dimensions_list, allow_unknown_extension=True)
    return sign_dimensions_list

def create_sign_dimensions_dict(sign_dimensions_protobuf):
    sign_dimensions_dict = {}
    for sign_dimension in sign_dimensions_protobuf.trackable_objs:
        sign_dimensions_dict[sign_dimension.type] = sign_dimension
    return sign_dimensions_dict

