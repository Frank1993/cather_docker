import math
import apollo_python_common.map_geometry.geometry_utils as geometry

class ProcessedSignPosition:

    def __init__(self, distance, angle_from_center, latitude, longitude, facing):
        self.distance = distance
        self.angle_from_center = angle_from_center
        self.latitude = latitude
        self.longitude = longitude
        self.facing = facing


class PositionCalculator:

    RAD2DEGFACTOR = 180 / math.pi
    EQUATORIAL_RADIUS = 6378137
    POLAR_RADIUS = 6356752
    EARTH_RADIUS_DELTA = EQUATORIAL_RADIUS - POLAR_RADIUS
    ANGLE_OF_ROI_MULTIPLIER = 90

    def __init__(self, phone_lens, camera_location, camera_heading, img_res, vp_x):
        self.phone_lens = phone_lens
        self.camera_location = camera_location
        self.camera_heading = camera_heading
        self.img_res = img_res
        self.vp_x = vp_x

    def calculate_sign_position(self, obj_bbox, obj_real_dim, angle_of_roi):
        horizontal_distance_to_obj_px = self.calculate_horizontal_distance_to_obj_px(obj_bbox)
        vertical_distance_to_obj_px = self.calculate_vertical_distance_to_obj_px(obj_bbox)
        horizontal_sensor_offset_mm = horizontal_distance_to_obj_px / self.img_res.width * self.phone_lens.sensor_width
        vertical_sensor_offset_mm = vertical_distance_to_obj_px / self.img_res.height * self.phone_lens.sensor_height
        horizontal_angle_from_center = self.calculate_angle_from_center(horizontal_sensor_offset_mm)
        vertical_angle_from_center = self.calculate_angle_from_center(vertical_sensor_offset_mm)
        distance_from_camera = self.calculate_distance_from_camera(obj_bbox, obj_real_dim, vertical_angle_from_center)
        latitude, longitude = self.calculate_position_offset_in_given_direction(
             self.camera_location.latitude, self.camera_location.longitude,
             self.camera_heading + horizontal_angle_from_center, distance_from_camera / 1000)
        facing = geometry.normalize_heading(self.camera_heading + angle_of_roi * self.ANGLE_OF_ROI_MULTIPLIER)
        position = ProcessedSignPosition(distance_from_camera, horizontal_angle_from_center, latitude, longitude, facing)

        return position

    def calculate_horizontal_distance_to_obj_px(self, obj_bbox):
        if self.vp_x is not None:
            horizontal_distance_to_obj_px = (obj_bbox.xmin + obj_bbox.xmax - 2 * self.vp_x) / 2
        else:
            horizontal_distance_to_obj_px = (obj_bbox.xmin + obj_bbox.xmax - self.img_res.width) / 2
        return horizontal_distance_to_obj_px

    def calculate_vertical_distance_to_obj_px(self, obj_bbox):
        vertical_distance_to_obj_px = (obj_bbox.ymin + obj_bbox.ymax - self.img_res.height) / 2
        return vertical_distance_to_obj_px

    def calculate_angle_from_center(self, sensor_offset_mm):
        angle_from_center = math.atan(sensor_offset_mm / self.phone_lens.focal_length) * self.RAD2DEGFACTOR
        return angle_from_center

    def calculate_distance_from_camera(self, obj_bbox, obj_real_dim, vertical_angle):
        distance_from_camera = (self.phone_lens.focal_length * obj_real_dim.height() * self.img_res.height)\
                               / (obj_bbox.height() * self.phone_lens.sensor_height)\
                               * math.cos(vertical_angle / self.RAD2DEGFACTOR)
        return distance_from_camera

    def calculate_position_offset_in_given_direction(self, latitude, longitude, heading, offset_m):
        sin_heading = math.sin(heading / self.RAD2DEGFACTOR)
        cos_heading = math.cos(heading / self.RAD2DEGFACTOR)
        latitude, longitude = self.calculate_position_offset(latitude, longitude,
                                                             offset_m * cos_heading, offset_m * sin_heading)
        return latitude, longitude

    def calculate_position_offset(self, latitude, longitude, lat_offset_m, lon_offset_m):
        cos_latitude = math.cos(latitude / self.RAD2DEGFACTOR)
        meridian_radius = self.POLAR_RADIUS + self.EARTH_RADIUS_DELTA * cos_latitude
        parallel_radius = self.EQUATORIAL_RADIUS * cos_latitude

        lat_offset_in_radians = lat_offset_m / meridian_radius
        lon_offset_in_radians = lon_offset_m / parallel_radius

        latitude += lat_offset_in_radians * self.RAD2DEGFACTOR
        longitude += lon_offset_in_radians * self.RAD2DEGFACTOR

        return latitude, longitude