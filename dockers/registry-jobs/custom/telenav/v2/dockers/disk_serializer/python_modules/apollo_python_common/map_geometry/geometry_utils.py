from math import radians, cos, sin, asin, sqrt, atan2, degrees
import numpy as np

EARTH_RADIUS = 6371000  # Radius of earth in meters


def compute_heading(start_latitude, start_longitude, end_latitude, end_longitude):
    start_latitude, start_longitude, end_latitude, end_longitude = map(radians,
                                                                       [start_latitude, start_longitude, end_latitude,
                                                                        end_longitude])
    delta_lon = end_longitude - start_longitude

    x = sin(delta_lon) * cos(end_latitude)
    y = cos(start_latitude) * sin(end_latitude) - sin(start_latitude) * cos(end_latitude) * cos(delta_lon)

    bearing = (degrees(atan2(x, y)) + 360) % 360  # bind bearing in the [0, 360) interval
    return bearing


def compute_haversine_distance(start_long, start_lat, end_long, end_lat):
    start_longitude, start_latitude, end_longitude, end_latitude = map(radians,
                                                                       [start_long, start_lat, end_long, end_lat])

    dlon = end_longitude - start_longitude
    dlat = end_latitude - start_latitude
    a = sin(dlat / 2) ** 2 + cos(start_latitude) * cos(end_latitude) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return c * EARTH_RADIUS


def normalize_heading(heading):
    if heading < 0:
        heading += 360
    elif heading > 360:
        heading -= 360
    return heading

def normalized_angle_difference(angle1, angle2):
    angle_diff = abs(angle1 - angle2)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    return angle_diff

def get_new_position(start_latitude, start_longitude, bearing, offset):
    start_longitude, start_latitude = map(radians, [start_longitude, start_latitude])
    bearing_rad = radians(bearing)

    lat2 = asin(sin(start_latitude) * cos(offset / EARTH_RADIUS) +
                cos(start_latitude) * sin(offset / EARTH_RADIUS) * cos(bearing_rad))

    lon2 = start_longitude + atan2(sin(bearing_rad) * sin(offset / EARTH_RADIUS) * cos(start_latitude),
                                   cos(offset / EARTH_RADIUS) - sin(start_latitude) * sin(lat2))

    return degrees(lat2), degrees(lon2)

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def get_angle(p0, p1, p2):
    ''' compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    '''
    if p2 is None:
        return 0.0
        # p2 = p1 + np.array([1, 0])

    if p1 is None:
        return 0.0

    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return np.degrees(angle)
