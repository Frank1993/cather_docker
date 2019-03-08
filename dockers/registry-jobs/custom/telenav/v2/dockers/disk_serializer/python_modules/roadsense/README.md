# Roadsense 

**General Info**

This module is responsible for analyzing OSC sensors in order to detect various hazards. The hazards are divided in two categories:
*   discrete: `speed bumps`, `big potholes`,`small potholes` and `sewer caps`
*   continuous: road quality, which can be `unpaved_road`,`bumpy_road` or `paved_road`

**A. Dataset Description**

The dataset has been constructed by performing test drives using multiple phones, each running the OSC app in order to record the sensors. In order to create the labeled data, a companion app named `HazardTracker` has been developed
which can be used in order to records the timestamp and GPS coordinates of the car when it hit a certain road hazard. The dataset, both for discrete and continuous data is grouped in separate folders, each folder holding the data for one day.
Each folder contains both the sensor data, which is situated in the `osc_sensor_data` subfolder and the hazard data, which is situated in the `hazard_data` subfolder

The dataset for discrete hazards can be found at 10.230.2.26:/data/bogdang/datasets/roadsense/part1/drives and for continuous hazards at 10.230.2.26:/data/bogdang/datasets/roadsense/part2/drives.

1. Hazard Data

    The hazard data is recorded in a json file having as components:
    * `car`
    * `driver`
    * `tagger`
    * list of `tags`. Each tag having a `location`, a `timestamp` and a `type`, which can be `speed_bump`,
    `big_pothole`,`small_pothole`,`sewer_cap`,`bumpy_road_start`,`bumpy_road_end` and `unpaved_road`
    
    For road quality, the following rules of tagging have been used:
    *   mark with `bumpy_road_start` and `bumpy_road_end` the points where the bumpy road starts/end.
    *   `unpaved road` class is marked by using `bumpy_road_start` and `bumpy_road_end` hazards but with an `unpaved_road` event between the two.
    
    
2. Sensor Data

    It is recorded in multiple csv files, each one representing the `metadata` for each trip recorded. There can be multiple csv files as there were multiple
    phones recording in the same time
    
    
**B. Dataset Preprocessing**

In the rest of this readme, we will use {MODULE_FOLDER} as a placeholder for `bump_detection` or `road_quality` folders, as name of the files in the two folders are identical  

In order to run the model training/prediction over the collected data, it needs to be processed such that representative features are being extracted and the hazard data is being merged with the sensor data. 
To run the preprocessing, you need to:

   * `cd python_modules/roadsense/run/{MODULE_FOLDER}/dataset`
   * fill `dataset_config.json` with appropriate values
   * `sh create_dataset.sh` 
   
   
JSON Parameter Description:

   * `dataset_base_path`: the path at which all the original dataset is stored, aka the subfolders for each day of data
   * `features_cols`: the sensor names which should be included as features 
   * `specific hazards`: the list of hazard types which should be added as labeled data
   * `blacklist_trips`: the csv filenames containing trip metadata which should be exlucded
   * `frequency`: the frequency at which the sensors should be samplled 
   * `steps`: how many steps should be included in every window
   * `hazard_buffer`: the number of steps at the beginning and end in every window which should be "hazard-free" 
   even though there is a hazard there in the data
   * `derived_window_size`: for the derived features, the number of steps which should be taken into consideration for computing each point
   * `scaler`: type of scaling to be used.
   * `drive_folders`: list of drive folders from `dataset_base_path` which should be included in the dataset
   * `phone_name`: filter for the device type, such that only trips from those phone types are included
   * `suffix`: identifier which will be added in the name of the preprocessed dataset folder 
   * `crop_start`:whether of not to crop the first 5% of the trip.
   * `add_match_data`:match metadata to way 
   * `with_custom_way_sections`: match each point to a precomputed ~100m way section

**C. Model Training**

In order to run the model training over the preprocessed dataset you need to:

   * `cd python_modules/roadsense/run/{MODULE_FOLDER}/train`
   * fill `train_config.json` with appropriate values
   * `sh train.sh` 
   
   
JSON Parameter Description:

   * `dataset_config_path`: the path to the config file which has been created by `train.sh`
   * `test_drive_days`: the days from the preprocessed dataset which should be used for testing
   * `bundle_path`: the output folder where the model weights/structure & train configuration should be saved 
   * `kept_hazards`: list of hazards which should be used from training (if you want not to use the whole hazards list which was used in the preprocessing step)
   * `batch_size`: model batch size
   * `epochs`: how many epochs to train
   * `ckpt_folder`: folder where to save the intermediary model checkpoints
   * `train_class_balance_factor`: number of non-hazard windows to include at train time compared to the number of hazard windows

**D. Prediction**

In order to run the prediction over the preprocessed dataset you need to:

   * `cd python_modules/roadsense/run/{MODULE_FOLDER}/predict`
   * fill `predict_config.json` with appropriate values
   * `sh predict.sh` 
      
JSON Parameter Description:   
      
   * `bundle_path`: the folder created by `train.sh`
   * `pred_input_folders`: the drive folder paths containing preprocessed data
   * `pred_output_folder`: path where the csv with the hazard detections should be generated

   
   