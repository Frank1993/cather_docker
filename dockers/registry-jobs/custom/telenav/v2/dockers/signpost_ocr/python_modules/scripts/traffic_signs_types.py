from apollo_python_common import proto_api as meta


class TrafficSignsTypes:
    def __init__(self, rois_file_name):
        self.rois_file_name = rois_file_name
        self.rois_dict = self._get_rois_dict_from_file_name()
        self.dict_type_idx = {type_id: idx for idx, type_id in enumerate(self._get_classes_from_rois_dict())}
        self.dict_idx_type = {v: k for k, v in self.dict_type_idx.items()}

    def _get_rois_dict_from_file_name(self):
        roi_metadata = meta.read_imageset_file(self.rois_file_name)
        rois_dict = meta.create_images_dictionary(roi_metadata)
        return rois_dict

    def _get_classes_from_rois_dict(self):
        return meta.get_classes_values_from_images_dictionary(self.rois_dict)

    def num_classes(self):
        return len(self.dict_idx_type)

    '''
    Gets the class index in the model based on class's type value (as in protobuf)
    '''
    def name_to_label(self, type):
        return self.dict_type_idx[type]

    '''
    Gets the class type numeric value based on class index as in the model
    '''
    def label_to_name(self, idx):
        return self.dict_idx_type[idx]

    def get_image_names(self):
        return list(self.rois_dict.keys())

    def _get_name_to_caption(self, type_value):
        return meta.get_roi_type_name(type_value)

    def labels(self):
        return list(self.dict_idx_type.values())

    def captions(self):
        return [self._get_name_to_caption(tv) for tv in self.dict_type_idx.keys()]

    def caption(self, type):
        return self._get_name_to_caption(type)

    '''
    Gets the class caption based on class index as in the model
    '''
    def label_to_caption(self, idx):
        return self.caption(self.label_to_name(idx))

    def caption_to_name(self, caption):
        return meta.get_roi_type_value(caption)
