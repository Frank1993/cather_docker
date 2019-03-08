import pandas as pd


def load_lidar_frame(frame_path):
    """ Loads the given lidar frame and returns a list of tuples with the points and their intensity.
            - x = tuple[0]
            - y = tuple[1]
            - z = tuple[2]
            - distance_m = tuple[3]
            - intensity = tuple[4]
    """
    frame_df = pd.read_csv(frame_path)
    points = []
    for idx, row in frame_df.iterrows():
        points.append((row['X'], row['Y'], row['Z'], row['distance_m'], row['intensity']))

    return points


if __name__ == '__main__':
    print('lidar frame: ', load_lidar_frame(
        '/Users/mihaic7/Development/projects/telenav/lidar/data/10_frames/10_frames (Frame 0000).csv'))