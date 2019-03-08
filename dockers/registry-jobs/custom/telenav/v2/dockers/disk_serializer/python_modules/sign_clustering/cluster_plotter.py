import argparse
import folium
from math import radians, cos, sin, asin, atan2, degrees
import pandas as pd
import random
from sklearn.metrics.cluster import adjusted_rand_score
import Geohash
from tqdm import tqdm
import math
from folium.plugins import MarkerCluster

import apollo_python_common.io_utils as io_utils
from apollo_python_common.map_geometry import geometry_utils as geom_utils
from sign_clustering import clustering
from sign_clustering.constants import *

def get_random_color():
    r = lambda: random.randint(0, 255)
    color = '#%02X%02X%02X' % (r(), r(), r())
    return color

def contruct_url(url, text):
    splits = text.split("\n")
    url = '<a href="' + url + '"target="_blank">'
    
    for s in splits:            
        url += s + "</br>"
        
    url += '</a>'
    return url

def draw_rois(data_df, status = None):
    
    view_map = folium.Map(tiles = 'cartodbpositron')
    
    status_2_color = {TP:'green',FP:'red',FN:'blue'}
    
    if status is not None:
        data_df = data_df[data_df[ROI_STATUS] == status]
        
    for _, row in tqdm(data_df.iterrows()):
        status = row[ROI_STATUS]
        roi_id = row[ROI_ID]
        if status == FN:
            lat, lon, heading, type_name = row[GT_LAT], row[GT_LON], row[GT_HEADING], row[GT_TYPE_NAME]
        else:
            lat, lon, heading, type_name = row[LAT], row[LON], row[HEADING], row[TYPE_NAME]
            
        text = f"Type {type_name} \n Heading {heading} \n Roi ID {roi_id} \n Status {status}"

        html_str = contruct_url(row[URL],text)
        popup = folium.Popup(html_str, max_width=300)

        folium.Circle([lat, lon], radius=3, popup=popup, fill=True, color=status_2_color[status]).add_to(view_map)
        heading_head_lat, heading_head_lon = geom_utils.get_new_position(lat, lon, heading, 7)
        folium.PolyLine(locations=[[lat, lon], [heading_head_lat, heading_head_lon]],
                        weight=2, color='black').add_to(view_map)

    path = "rois.html"            
    view_map.fit_bounds(view_map.get_bounds())
    view_map.save(path)
    
    return path

def draw_gt_clusters(gt_clusters_df,pred_clusters_df,invalid_cluster_pred_df,data_df, status_list = None, view_map = None):
    
    if view_map is None:
        view_map = folium.Map(tiles = 'cartodbpositron')
    
    roi_2_pred_data = dict()

    for _, r in pred_clusters_df.iterrows():
        pred_cluster = r[PRED_CLUSTER_ID]
        matched_gt_cluster = r[MATCHED_GT_CLUSTER_ID]

        for roi_id in r[MEMBER_ROIS]:
            roi_2_pred_data[roi_id] = (pred_cluster,matched_gt_cluster)


    for _, r in invalid_cluster_pred_df.iterrows():
        for roi_id in r[MEMBER_ROIS]:
            roi_2_pred_data[roi_id] = (INVALID_CLUSTER_ID,INVALID_CLUSTER_ID)


    roi_data = data_df.set_index(ROI_ID).to_dict('index')
    
    status_2_color = {TP:'green', FN:'blue'}
    
    if status_list is not None:
        gt_clusters_df = gt_clusters_df[gt_clusters_df[CLUSTER_STATUS].isin(status_list)]
        
    for _, r in tqdm(gt_clusters_df.iterrows()):
        cluster_status = r[CLUSTER_STATUS]
        gt_cluster_id = r[GT_CLUSTER_ID]
        lat, lon, heading, type_name = r[GT_LAT], r[GT_LON], round(r[GT_HEADING],2), r[GT_TYPE_NAME]
        cluster_status_reason = r[CLUSTER_STATUS_REASON]
        
        if math.isnan(r[MATCHED_PRED_CLUSTER_ID]):
            matched_pred_cluster_id_text= f"No matched pred cluster"
        else: 
            matched_pred_cluster_id_text = f"Matched pred cluster {int(r[MATCHED_PRED_CLUSTER_ID])}"
        
        text = f"Type {type_name} \n Heading {heading} \n GT cluster ID {gt_cluster_id} \n Cluster Status {cluster_status} \nCluster Status Reason {cluster_status_reason} \n {matched_pred_cluster_id_text}"
        html_str = contruct_url("",text)
        
        for roi_id in r[MEMBER_ROIS]:
            pred_cluster,matched_gt_cluster = roi_2_pred_data[roi_id]
            url = roi_data[roi_id][URL]
            pred_heading = round(roi_data[roi_id][HEADING],2)
            roi_status = roi_data[roi_id][ROI_STATUS]
            cluster_outcome = roi_data[roi_id][CLUSTER_OUTCOME]
            
            if pred_cluster != INVALID_CLUSTER_ID:
                pred_cluster_text =  f"PC {pred_cluster}"
                matched_gt_cluster_text = f"MGTC {matched_gt_cluster}"
            else:
                pred_cluster_text,matched_gt_cluster_text = "N/A","N/A"

            text = f"\n{roi_id} -> {pred_cluster_text} -> {matched_gt_cluster_text}"
            text += f"\nPred heading {pred_heading}\n ROI Status {roi_status} \nCluster Outcome {cluster_outcome}"
            
            
            html_str += contruct_url(url,text)
            
        popup = folium.Popup(html_str, max_width=300)

        folium.Circle([lat, lon], radius=3, popup=popup, fill=True, color=status_2_color[cluster_status]).add_to(view_map)
        heading_head_lat, heading_head_lon = geom_utils.get_new_position(lat, lon, heading, 7)
        folium.PolyLine(locations=[[lat, lon], [heading_head_lat, heading_head_lon]],
                        weight=2, color='black').add_to(view_map)
    
    return view_map

def draw_pred_clusters(gt_clusters_df,pred_clusters_df,invalid_cluster_pred_df,data_df, status_list = None, view_map = None):
    
    if view_map is None:
        view_map = folium.Map(tiles = 'cartodbpositron')
        
    roi_2_gt_data = dict()

    for _, r in gt_clusters_df.iterrows():
        gt_cluster = r[GT_CLUSTER_ID]
        matched_pred_cluster = r[MATCHED_PRED_CLUSTER_ID]

        for roi_id in r[MEMBER_ROIS]:
            roi_2_gt_data[roi_id] = (gt_cluster,matched_pred_cluster)


    for _, r in invalid_cluster_pred_df.iterrows():
        for roi_id in r[MEMBER_ROIS]:
            roi_2_gt_data[roi_id] = (INVALID_CLUSTER_ID,INVALID_CLUSTER_ID)


    roi_data = data_df.set_index(ROI_ID).to_dict('index')
    
    status_2_color = {TP:'green', FP:'red'}
    
    if status_list is not None:
        pred_clusters_df = pred_clusters_df[pred_clusters_df[CLUSTER_STATUS].isin(status_list)]
        
    for _, r in tqdm(pred_clusters_df.iterrows()):
        cluster_status = r[CLUSTER_STATUS]
        pred_cluster_id = r[PRED_CLUSTER_ID]
        lat, lon, heading, type_name = r[LAT], r[LON], round(r[HEADING],2), r[TYPE_NAME]
        cluster_status_reason = r[CLUSTER_STATUS_REASON]
        
        if math.isnan(r[MATCHED_GT_CLUSTER_ID]):
            matched_gt_cluster_id_text= f"No matched GT cluster"
        else: 
            matched_gt_cluster_id_text = f"Matched GT cluster {int(r[MATCHED_GT_CLUSTER_ID])}"
        
        text = f"Type {type_name} \n Heading {heading} \n Pred Cluster ID {pred_cluster_id} \n Cluster status {cluster_status} \nCluster Status Reason {cluster_status_reason} \n {matched_gt_cluster_id_text}"
        html_str = contruct_url("",text)
        
        for roi_id in r[MEMBER_ROIS]:
            roi_status = roi_data[roi_id][ROI_STATUS]
            if roi_id in roi_2_gt_data:
                gt_cluster,matched_pred_cluster = roi_2_gt_data[roi_id]
            else:
                gt_cluster,matched_pred_cluster = "N/A","N/A"
                
            url = roi_data[roi_id][URL]
            pred_heading = round(roi_data[roi_id][HEADING],2)
            
            gt_cluster_text =  f"GTC {gt_cluster}"
            cluster_outcome = roi_data[roi_id][CLUSTER_OUTCOME]
            
            text = f"\n{roi_id} -> {gt_cluster_text}"
            text += f"\nPred heading {pred_heading}\n ROI Status {roi_status}"
            
            html_str += contruct_url(url,text)
            
        popup = folium.Popup(html_str, max_width=300)

        folium.Circle([lat, lon], radius=3, popup=popup, fill=True, color=status_2_color[cluster_status]).add_to(view_map)
        heading_head_lat, heading_head_lon = geom_utils.get_new_position(lat, lon, heading, 7)
        folium.PolyLine(locations=[[lat, lon], [heading_head_lat, heading_head_lon]],
                        weight=2, color='black').add_to(view_map)


    return view_map

def draw_clusters(gt_clusters_df,pred_clusters_df,invalid_cluster_pred_df,data_df, status_list = None):
    view_map = folium.Map(tiles = 'cartodbpositron')
        
    if status_list is None:
        status_list = [TP,FP,FN]
        
    pred_status_list = {TP,FP}.intersection(set(status_list))
    gt_status_list = {FN}.intersection(set(status_list))
   
    view_map = draw_pred_clusters(gt_clusters_df,pred_clusters_df,invalid_cluster_pred_df,data_df, 
                                       status_list = pred_status_list, view_map = view_map)

    view_map = draw_gt_clusters(gt_clusters_df,pred_clusters_df,invalid_cluster_pred_df,data_df, 
                                       status_list = gt_status_list, view_map = view_map)
        
    path = "clusters.html"            
    view_map.fit_bounds(view_map.get_bounds())
    view_map.save(path)

    return path
    
def draw_pred_rois_clustered(data_df, status = None):
    
    view_map = folium.Map(tiles = 'cartodbpositron')
    
    idx_2_color = {i: get_random_color() for i in data_df[PRED_CLUSTER_ID].unique()}
    idx_2_color[INVALID_CLUSTER_ID] = "cyan"
    
    data_df = data_df[data_df[ROI_STATUS] != FN]
    
    if status is not None:
        data_df = data_df[data_df[ROI_STATUS] == status]
        
    for _, row in tqdm(data_df.iterrows()):
        status = row[ROI_STATUS]
        roi_id = row[ROI_ID]
        pred_cluster_id = row[PRED_CLUSTER_ID]
        gt_cluster_id = row[GT_CLUSTER_ID]
        cluster_outcome = row[CLUSTER_OUTCOME]
        
        lat, lon, heading, type_name = row[LAT], row[LON], row[HEADING], row[TYPE_NAME]
            
        text = f"Type {type_name} \nHeading {heading} \nRoi ID {roi_id} \nStatus {status} \
        \nGT Cluster {gt_cluster_id} \nPred Cluster {pred_cluster_id} \nCluster Outcome {cluster_outcome}"

        html_str = contruct_url(row[URL],text)
        popup = folium.Popup(html_str, max_width=300)
        
        if math.isnan(pred_cluster_id):
            print(row)
        folium.Circle([lat, lon], radius=3, popup=popup, fill=True, color=idx_2_color[pred_cluster_id]).add_to(view_map)
        heading_head_lat, heading_head_lon = geom_utils.get_new_position(lat, lon, heading, 7)
        folium.PolyLine(locations=[[lat, lon], [heading_head_lat, heading_head_lon]],
                        weight=2, color='black').add_to(view_map)

    path = "pred_rois.html"            
    view_map.fit_bounds(view_map.get_bounds())
    view_map.save(path)
    
    return path


def draw_gt_rois_clustered(data_df, status = None):
    
    view_map = folium.Map(tiles = 'cartodbpositron')
    
    idx_2_color = {i: get_random_color() for i in data_df[GT_CLUSTER_ID].unique()}
    
    data_df = data_df[data_df[ROI_STATUS] != FP]
    
    if status is not None:
        data_df = data_df[data_df[ROI_STATUS] == status]
        
    for _, row in tqdm(data_df.iterrows()):
        status = row[ROI_STATUS]
        roi_id = row[ROI_ID]
        pred_cluster_id = row[PRED_CLUSTER_ID]
        gt_cluster_id = row[GT_CLUSTER_ID]
        cluster_outcome = row[CLUSTER_OUTCOME]
        
        lat, lon, heading, type_name = row[GT_LAT], row[GT_LON], row[GT_HEADING], row[GT_TYPE_NAME]
            
        text = f"Type {type_name} \nHeading {heading} \nRoi ID {roi_id} \nStatus {status} \
        \nGT Cluster {gt_cluster_id} \nPred Cluster {pred_cluster_id} \nCluster Outcome {cluster_outcome}"

        html_str = contruct_url(row[URL],text)
        popup = folium.Popup(html_str, max_width=300)

        folium.Circle([lat, lon], radius=3, popup=popup, fill=True, color=idx_2_color[gt_cluster_id]).add_to(view_map)
        heading_head_lat, heading_head_lon = geom_utils.get_new_position(lat, lon, heading, 7)
        folium.PolyLine(locations=[[lat, lon], [heading_head_lat, heading_head_lon]],
                        weight=2, color='black').add_to(view_map)

    path = "gt_rois.html"            
    view_map.fit_bounds(view_map.get_bounds())
    view_map.save(path)
    
    return path
