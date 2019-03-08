from apollo_python_common.rectangle import Rectangle
from sign_positioning.mq.generic_sign_position_calculator import GenericSignPositionCalculator


class CombinedSpeedLimitPositionCalculator(GenericSignPositionCalculator):

    VERTICAL_PX_OFFSET = 20
    COMBINED_SL_HEIGHT = 381

    def __init__(self, phone_lens, camera_location, camera_heading, img_res, vp_x, rois):
        super().__init__(phone_lens, camera_location, camera_heading, img_res, vp_x)
        self.rois = rois

    def is_combined_speed_limit(self, combined_sl_roi):
        for roi in self.rois:
            roi_br_row = roi.rect.br.row
            sl_roi_tl_row = combined_sl_roi.rect.tl.row
            if sl_roi_tl_row - self.VERTICAL_PX_OFFSET < roi_br_row:
                return True

    def calculate_sign_position(self, roi, obj_dimensions, angle_of_roi, highway_type):
        obj_bbox = Rectangle(roi.rect.tl.col, roi.rect.tl.row, roi.rect.br.col, roi.rect.br.row)
        if self.is_combined_speed_limit(roi):
            obj_dimensions.height = self.COMBINED_SL_HEIGHT
        obj_real_dim = super().adjust_sign_size(obj_dimensions, highway_type)
        return self.position_calculator.calculate_sign_position(obj_bbox, obj_real_dim, angle_of_roi)

