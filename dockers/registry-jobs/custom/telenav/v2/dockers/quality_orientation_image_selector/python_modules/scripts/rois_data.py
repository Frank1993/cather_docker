from apollo_python_common import proto_api as meta


class RoisLabels:
    def __init__(self,
                 rois_file_name):
        self.rois_file_name=rois_file_name
        self.rois_dict = self.__get_rois_dict_from_file_name()
        self.classes = dict([(class_name, id) for id, class_name in enumerate(self.__get_classes_from_rois_dict())])
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __get_rois_dict_from_file_name(self):
        roi_metadata = meta.read_imageset_file(self.rois_file_name)
        rois_dict = meta.create_images_dictionary(roi_metadata)
        return rois_dict

    def __get_classes_from_rois_dict(self):
        return meta.get_class_names_from_images_dictionary(self.rois_dict)

    def num_classes(self):
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def get_image_names(self):
        return [filename for filename in self.rois_dict.keys()]
