import os
from glob import glob
import pandas as pd
from tqdm import tqdm

from roadsense.scripts.config import ConfigParams as cp, HazardType as HazardType


def get_sensor_df_id(sensor_path):
    with open(sensor_path) as file:
        phone_type = "-".join(file.readline().split(";")[0].split(" "))

    username = os.path.basename(sensor_path).split("_")[0]
    trip_id = sensor_path.split("_")[-1].split(".")[0]

    return f"{username}_{phone_type}_{trip_id}"

def read_sensor_df(input_path):
    paths = glob(input_path+"/*")
    return pd.concat([pd.read_pickle(path) for path in tqdm(paths)])

def get_dataset_folder_name(config):
    return "f={}_s={}_b={}_pn={}_s={}_s={}".format(config[cp.FREQUENCY],
                                                config[cp.STEPS],
                                                config[cp.HAZARD_BUFFER_STEPS],
                                                config[cp.PHONE_NAME],
                                                config[cp.SCALER_TYPE],
                                                config[cp.SUFFIX]
                                               )

def keep_high_conf_hazards(y_pred_proba,conf_thresh, class_2_index):
    #todo add support for multiple hazards
    hazard_of_interest  = list(set(class_2_index.keys()) - set([HazardType.CLEAR]))[0]

    return [hazard_of_interest if pred_arr[class_2_index[hazard_of_interest]] > conf_thresh
              else HazardType.CLEAR for pred_arr in y_pred_proba]