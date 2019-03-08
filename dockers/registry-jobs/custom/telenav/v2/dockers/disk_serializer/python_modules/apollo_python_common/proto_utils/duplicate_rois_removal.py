import pandas as pd
from tqdm import tqdm

tqdm.pandas()

from apollo_python_common.rectangle import Rectangle
import orbb_metadata_pb2

IMAGE_PROTO, ROIS, IMG_NAME, TO_KEEP, TO_DELETE = ["image_proto", "rois", "img_name", "to_keep", "to_delete"]

OVERLAP_THRESHOLD = 0.75


def read_df(imageset_proto):
    image_proto_list = list(imageset_proto.images)

    det_df = pd.DataFrame({IMAGE_PROTO: image_proto_list})
    det_df.loc[:, ROIS] = det_df.loc[:, IMAGE_PROTO].apply(lambda ip: ip.rois)
    det_df.loc[:, IMG_NAME] = det_df.loc[:, IMAGE_PROTO].apply(lambda ip: ip.metadata.image_path)

    return det_df


def are_rois_intersecting(roi_1, roi_2):
    rect_1 = roi_1.rect
    rect_2 = roi_2.rect

    rect_1 = Rectangle(rect_1.tl.col, rect_1.tl.row, rect_1.br.col, rect_1.br.row)
    rect_2 = Rectangle(rect_2.tl.col, rect_2.tl.row, rect_2.br.col, rect_2.br.row)

    return rect_1.intersection_over_union(rect_2) > OVERLAP_THRESHOLD and roi_1.type == roi_2.type


def is_roi_intersecting_list(roi, target_list):
    for target_roi in target_list:
        if are_rois_intersecting(roi, target_roi):
            return True

    return False


def order_by_type_and_confidence(roi):  # if manual, should be first in list. else, order by confidence
    if roi.manual:
        return 2  # random number bigger than 1
    else:
        return roi.detections[0].confidence


def get_delete_roi_list(row):
    rois = sorted(row[ROIS], key=order_by_type_and_confidence)

    rois_to_keep = []
    rois_to_delete = []

    for roi in rois:
        is_intersecting = is_roi_intersecting_list(roi, rois_to_keep)

        if is_intersecting:
            rois_to_delete.append(roi)
        else:
            rois_to_keep.append(roi)

    return rois_to_delete


def df_2_proto(rois_df):
    imageset_proto = orbb_metadata_pb2.ImageSet()
    imageset_proto.name = "imageset"

    for _, row in rois_df.iterrows():
        image_proto = row[IMAGE_PROTO]
        rois_list = image_proto.rois
        to_delete = row[TO_DELETE]

        for roi_to_delete in to_delete:
            rois_list.remove(roi_to_delete)

        imageset_proto.images.extend([image_proto])

    return imageset_proto


def remove_duplicate_rois(imageset_proto):
    rois_df = read_df(imageset_proto)

    rois_df.loc[:, TO_DELETE] = rois_df.progress_apply(get_delete_roi_list, axis=1)

    return df_2_proto(rois_df)

