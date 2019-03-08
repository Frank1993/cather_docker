import os
from glob import glob
import pandas as pd
from tqdm import tqdm

from roadsense.scripts.general.config import ConfigParams as cp, HazardType as HazardType


def get_sensor_df_id(sensor_path,reader):
    phone_type = reader.get_device_name(sensor_path)
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

def keep_high_conf_hazards(y_pred_proba, hazard_2_best_conf, class_2_index):
    hazards_by_impact = [HazardType.UNPAVED_ROAD,
                         HazardType.BUMPY_ROAD,
                         HazardType.SPEED_BUMP,
                         HazardType.BIG_POTHOLE,
                         HazardType.SMALL_POTHOLE,
                         HazardType.SEWER_HOLE]
        
    hazards_by_impact = [h for h in hazards_by_impact if h in hazard_2_best_conf]

    y_pred = []
    for pred_arr in y_pred_proba:
        chosen_class = HazardType.CLEAR

        for hazard in hazards_by_impact:
            pred_conf = pred_arr[class_2_index[hazard]]
            if pred_conf > hazard_2_best_conf[hazard]:
                chosen_class = hazard
                break

        y_pred.append(chosen_class)
        
        
    return y_pred