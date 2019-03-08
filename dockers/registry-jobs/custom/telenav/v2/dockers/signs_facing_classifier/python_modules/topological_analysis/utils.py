from math import radians, cos, sin, asin, sqrt, atan2, degrees


def compute_haversine_distance(start, end):
    start_longitude, start_latitude, end_longitude, end_latitude = map(radians,
                                                                       [start.longitude, start.latitude, end.longitude,
                                                                        end.latitude])

    dlon = end_longitude - start_longitude
    dlat = end_latitude - start_latitude
    a = sin(dlat / 2) ** 2 + cos(start_latitude) * cos(end_latitude) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 000  # Radius of earth in meters
    return c * r


def compute_bearing(start, end):
    start_longitude, start_latitude, end_longitude, end_latitude = map(radians,
                                                                       [start.longitude, start.latitude, end.longitude,
                                                                        end.latitude])
    delta_lon = end_longitude - start_longitude

    x = sin(delta_lon) * cos(end_latitude)
    y = cos(start_latitude) * sin(end_latitude) - sin(start_latitude) * cos(end_latitude) * cos(delta_lon)

    bearing = (degrees(atan2(x, y)) + 360) % 360  # bind bearing in the [0, 360) interval

    return bearing


def revert_heading(heading):
    return (heading + 180) % 360
