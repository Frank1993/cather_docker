import itertools
import random

import folium
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from roadsense.scripts.config import ConfigParams as cp, HazardType as HazardType, Column

hazard_2_color = {
    "speed_bump": (1, 0, 1),
    "sewer_hole": (0, 1, 0),
    "small_pothole": (1, 1, 0),
    "big_pothole": (0, 1, 1)
}


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')

    plt.show()


def plt_confusion_matrix(y_test, y_pred):
    cnf_matrix = confusion_matrix(y_pred, y_test)
    plot_confusion_matrix(cnf_matrix, classes=sorted(set(y_test)))


def plot_merged_feature_2_label(merged_df, feature_col_name, start_time=None, end_time=None):
    if start_time is not None and end_time is not None:
        plt_merged_df = merged_df.loc[start_time:end_time, :]
    else:
        plt_merged_df = merged_df

    plt.figure(figsize=(20, 5))
    plt.title(feature_col_name)

    plt_merged_df[feature_col_name].plot()

    for datetime, row in plt_merged_df.iterrows():
        if row[Column.HAZARD] != HazardType.CLEAR:
            color = hazard_2_color[row[Column.HAZARD]]
            plt.axvline(datetime, color=color, alpha=0.3)

    plt.show()


def plot_merged_feature_2_label_combined(sensor_df_list, feature_col_name, start_time=None, end_time=None,
                                         with_hazards=True):
    plt.figure(figsize=(20, 5))
    plt.title(feature_col_name)

    for index, sensor_df in enumerate(sensor_df_list):

        plt_sensor_df = sensor_df.copy()

        sensor_id = sensor_df[Column.TRIP_ID].tolist()[0]

        plt_sensor_df = plt_sensor_df.loc[start_time:end_time, :] if start_time is not None else sensor_df
        plt_sensor_df = plt_sensor_df.dropna(subset=[feature_col_name])

        new_col_name = sensor_id
        plt_sensor_df[new_col_name] = plt_sensor_df[feature_col_name]
        plt_sensor_df[new_col_name].plot()

        if with_hazards and index == 0:  # just draw them for the first df, all the same
            for datetime, row in plt_sensor_df.iterrows():
                if row[Column.HAZARD] != HazardType.CLEAR:
                    color = hazard_2_color[row[Column.HAZARD]]
                    plt.axvline(datetime, color=color, alpha=0.3)

    plt.legend()
    plt.show()


def plot_confidence_metrics(conf_levels, precisions, recalls, f_scores):
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(121)

    plt.plot(conf_levels, precisions, label='precision')
    for i, j in zip(conf_levels, precisions):
        ax.annotate(str(int(round(j, 2) * 100)), xy=(i, j))

    plt.plot(conf_levels, recalls, label='recall')
    for i, j in zip(conf_levels, recalls):
        ax.annotate(str(int(round(j, 2) * 100)), xy=(i, j))

    plt.plot(conf_levels, f_scores, label='f_score')
    for i, j in zip(conf_levels, f_scores):
        ax.annotate(str(int(round(j, 2) * 100)), xy=(i, j))

    plt.legend(loc='best')
    plt.show()

    for conf_level, f_score, prec, recall in zip(conf_levels, f_scores, precisions, recalls):
        print("C= %.2f ---> F= %.2f / P= %.2f / R= %.2f" % (conf_level, f_score, prec, recall))


def get_random_color():
    r = lambda: random.randint(0, 255)
    color = '#%02X%02X%02X' % (r(), r(), r())
    return color


def plot_windows(clustered_df_original, centroid_df, lat_col, lon_col, nr_items=None, with_centroid=True):
    clustered_df = clustered_df_original.drop_duplicates([Column.LAT, Column.LON])

    view_map = folium.Map()
    idx_2_color = {i: get_random_color() for i in centroid_df[Column.CLUSTER_ID].unique()}

    if nr_items is None:
        nr_items = len(clustered_df)

    for _, row in tqdm(clustered_df[:nr_items].iterrows()):
        trip_id = row["trip_id"].split("_")[-1]
        popup_text = "www.openstreetcam.org/details/{}/{}/".format(trip_id, int(row[Column.IMAGE_INDEX]))
         
        color = idx_2_color[row[Column.CLUSTER_ID]] if row[Column.CLUSTER_ID] in idx_2_color else 'black'

        hazard_lat, hazard_lon = float(row[lat_col]), float(row[lon_col])
        hazard_point = folium.Circle([hazard_lat, hazard_lon],popup=folium.Popup(popup_text),
                                     radius=3, color=color, fill=True, fill_color=color)
        view_map.add_child(hazard_point)

        centroid_row = centroid_df[centroid_df[Column.CLUSTER_ID] == row[Column.CLUSTER_ID]].iloc[0]
        centroid_lat, centroid_lon = centroid_row[Column.CLUSTER_LAT],centroid_row[Column.CLUSTER_LON]
                                   
        centroid_point = folium.Circle([centroid_lat, centroid_lon], popup=folium.Popup(popup_text),
                                       radius=10, color=color, fill=True, fill_color=color)
        view_map.add_child(centroid_point)

    return view_map


def add_gt_centroids(view_map, clustered_df, centroid_df, lat_col, lon_col):
    for _, row in tqdm(clustered_df.iterrows()):
        
        centroid_row = centroid_df[centroid_df[Column.CLUSTER_ID] == row[Column.CLUSTER_ID]].iloc[0]
        centroid_lat, centroid_lon = centroid_row[Column.CLUSTER_LAT],centroid_row[Column.CLUSTER_LON]

        centroid_point = folium.Circle([centroid_lat, centroid_lon], popup="{},{} \n {} {}".format(
            row[Column.HAZARD_LAT],
            row[Column.HAZARD_LON],
            row[Column.CLUSTER_ID], row[Column.DATETIME]),
                                       radius=10, color='cyan', fill=True, fill_color='cyan')

        view_map.add_child(centroid_point)

    return view_map


def plot_matchings(tp_list, fp_list, fn_list, pred_idx_2_centroid, gt_idx_2_centroid, view_map=None):
    # Green TP
    # Blue FP
    # Red FN

    if view_map is None:
        view_map = folium.Map()

    for gt_id, pred_id in tp_list:
        gt_centroid = gt_idx_2_centroid[gt_id]
        pred_centroid = pred_idx_2_centroid[pred_id]

        gt_point = folium.Circle([gt_centroid[0], gt_centroid[1]], radius=3, color='green',
                                 fill=True, fill_color='green')

        pred_point = folium.Circle([pred_centroid[0], pred_centroid[1]], radius=3, color='green',
                                   fill=True, fill_color='green')

        polyline = folium.PolyLine(locations=[[gt_centroid[0], gt_centroid[1]], [pred_centroid[0], pred_centroid[1]]],
                                   weight=4, color="green")

        view_map.add_child(pred_point)
        view_map.add_child(gt_point)
        view_map.add_child(polyline)

    for gt_id in fn_list:
        gt_centroid = gt_idx_2_centroid[gt_id]
        gt_point = folium.Circle([gt_centroid[0], gt_centroid[1]], radius=3, color='red',
                                 fill=True, fill_color='red')
        view_map.add_child(gt_point)

    for pred_id in fp_list:
        pred_centroid = pred_idx_2_centroid[pred_id]
        pred_point = folium.Circle([pred_centroid[0], pred_centroid[1]], radius=3, color='blue',
                                   fill=True, fill_color='blue')

        view_map.add_child(pred_point)

    return view_map


def plot_meta_clusters(meta_clustered_df, meta_centroid_df, nr_items=None, is_hazard=False):
    view_map = folium.Map()
    idx_2_color = {i: get_random_color() for i in meta_centroid_df["meta_cluster_id"].unique()}

    if nr_items is None:
        nr_items = len(meta_clustered_df)

    for _, row in tqdm(meta_clustered_df[:nr_items].iterrows()):

        color = idx_2_color[row["meta_cluster_id"]] if row["meta_cluster_id"] in idx_2_color else 'black'

        if is_hazard:
            color = 'cyan'

        hazard_lat, hazard_lon = float(row["meta_cluster_lat"]), float(row["meta_cluster_lon"])
        hazard_point = folium.Circle([hazard_lat, hazard_lon], radius=3, popup = str(row["meta_cluster_id"]),
                                     color=color, fill=True, fill_color=color)
        view_map.add_child(hazard_point)

        centroid_row = meta_centroid_df[meta_centroid_df["meta_cluster_id"] == row["meta_cluster_id"]].iloc[0]
        centroid_lat, centroid_lon = centroid_row["meta_cluster_lat"],centroid_row["meta_cluster_lon"]

        centroid_point = folium.Circle([centroid_lat, centroid_lon], radius=10,popup = str(row["meta_cluster_id"]),
                                       color=color, fill=True,
                                       fill_color=color)
        view_map.add_child(centroid_point)

    return view_map
