from apollo_python_common.rectangle import Rectangle
from sign_positioning.mq.generic_sign_position_calculator import GenericSignPositionCalculator


class TrafficLightPositionCalculator(GenericSignPositionCalculator):

    def __init__(self, phone_lens, camera_location, camera_heading, img_res, vp_x):
        super().__init__(phone_lens, camera_location, camera_heading, img_res, vp_x)

    def calculate_sign_position(self, roi, obj_dimensions, angle_of_roi, highway_type):
        obj_bbox = Rectangle(roi.rect.tl.col, roi.rect.tl.row, roi.rect.br.col, roi.rect.br.row)
        obj_real_dim = super().adjust_sign_size(obj_dimensions, highway_type)
        if obj_real_dim.height() < obj_real_dim.width():
            obj_real_dim = Rectangle(0, 0, obj_real_dim.height, obj_real_dim.width)
        return self.position_calculator.calculate_sign_position(obj_bbox, obj_real_dim, angle_of_roi)