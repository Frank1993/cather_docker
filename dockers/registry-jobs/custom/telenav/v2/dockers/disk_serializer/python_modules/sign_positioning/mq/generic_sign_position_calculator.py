from apollo_python_common.rectangle import Rectangle
from sign_positioning.mq.position_calculator import PositionCalculator


class GenericSignPositionCalculator:

    def __init__(self, phone_lens, camera_location, camera_heading, img_res, vp_x):
        self.position_calculator = PositionCalculator(phone_lens, camera_location, camera_heading, img_res, vp_x)

    def calculate_sign_position(self, roi, obj_dimensions, angle_of_roi, highway_type):
        obj_bbox = Rectangle(roi.rect.tl.col, roi.rect.tl.row, roi.rect.br.col, roi.rect.br.row)
        obj_real_dim = self.adjust_sign_size(obj_dimensions, highway_type)
        return self.position_calculator.calculate_sign_position(obj_bbox, obj_real_dim, angle_of_roi)

    @staticmethod
    def adjust_sign_size( obj_dimensions, highway_type):
        width = obj_dimensions.width
        height = obj_dimensions.height
        if "motorway" == highway_type:
            width *= obj_dimensions.motorway_size_increase
            height *= obj_dimensions.motorway_size_increase
        elif "trunk" == highway_type:
            width *= obj_dimensions.trunk_size_increase
            height *= obj_dimensions.trunk_size_increase
        return Rectangle(0, 0, width, height)