from apollo_python_common import proto_api as proto_api
from scripts.rois_data import RoisLabels


class SignElementsLabels(RoisLabels):
    def __init__(self, rois_file_name):
        self.rois_file_name = rois_file_name
        roi_metadata = proto_api.read_imageset_file(self.rois_file_name)
        self.rois_dict = proto_api.create_images_dictionary(roi_metadata)
        self.classes = {class_name: id for id, class_name in enumerate(self.__get_classes_from_rois_dict())}
        self.labels = {v: k for k, v in self.classes.items()}

    def __get_classes_from_rois_dict(self):
        return proto_api.get_class_names_from_images_dictionary(self.rois_dict, translate_to_sign_components=True)
