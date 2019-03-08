# Roadsense 

**General Info**

This module is responsible for analyzing OSC sensors in order to detect various hazards such as: speed bumps, potholes and sewer caps  


**A. Dataset Description**

The dataset has been constructed by performing test drives using multiple phones, each running the OSC app in order to record the sensors. In order to create the labeled data, a companion app named `HazardTracker` has been developed
which can be used in order to records the timestamp and GPS coordinates of the car when it hit a certain road hazard 

The dataset can be found at 10.230.2.26:/data/bogdang/datasets/roadsense/drives. It's grouped in separate folders, each folder holding the data for one day.
Each folder contains both the sensor data, which is situated in the `osc_sensor_data` subfolder and the hazard data, which is situated in the `hazard_data` subfolder


1. Hazard Data

    The hazard data is recorded in a json file having as components:
    * `car`
    * `driver`
    * `tagger`
    * list of `tags`. Each tag having a `location`, a `timestamp` and a `type`, which can be `speed_bump`,
    `big_pothole`,`small_pothole` and `sewer_cap`

2. Sensor Data

    It is recorded in multiple csv files, each one representing the `metadata` for each trip recorded. There can be multiple csv files as there were multiple
    phones recording in the same time
    
    
**B. Dataset Preprocessing**


In order to run the model training/prediction over the collected data, it needs to be processed such that representative features are being extracted and the hazard data is being merged with the sensor data. 
To run the preprocessing, you need to:

   * `cd python_modules/roadsense/run/dataset`
   * fill `dataset_config.json` with appropriate values
   * `sh create_dataset.sh` 
   
   
JSON Parameter Description:

   * `dataset_base_path`: the path at which all the original dataset is stored, aka the subfolders for each day of data
   * `features_cols`: the sensor names which should be included as features 
   * `specific hazards`: the list of hazard types which should be added as labeled data
   * `blacklist_trips`: the csv filenames containing trip metadata which should be exlucded
   * `FREQUENCY`: the frequency at which the sensors should be samplled 
   * `steps`: how many steps should be included in every window
   * `hazard_buffer`: the number of steps at the beginning and end in every window which should be "hazard-free" 
   even though there is a hazard there in the data
   * `derived_window_size`: for the derived features, the number of steps which should be taken into consideration for computing each point
   * `scaler`: type of scaling to be used.
   * `drive_folders`: list of drive folders from `dataset_base_path` which should be included in the dataset
   * `phone_name`: filter for the device type, such that only trips from those phone types are included
   * `suffix`: identifier which will be added in the name of the preprocessed dataset folder 
   
**C. Model Training**

In order to run the model training over the preprocessed dataset you need to:

   * `cd python_modules/roadsense/run/train`
   * fill `train_config.json` with appropriate values
   * `sh train.sh` 
   
   
JSON Parameter Description:

   * `dataset_config_path`: the path to the config file which has been created by `train.sh`
   * `test_drive_day`: the day from the preprocessed dataset which should be used for testing
   * `bundle_path`: the output folder where the model weights/structure & train configuration should be saved 
   * `kept_hazards`: list of hazards which should be used from training (if you want not to use the whole hazards list which was used in the preprocessing step)
   * `batch_size`: model batch size
   * `epochs`: how many epochs to train
   * `ckpt_folder`: folder where to save the intermediary model checkpoints
   * `train_class_balance_factor`: number of non-hazard windows to include at train time compared to the number of hazard windows

**D. Prediction**

In order to run the prediction over the preprocessed dataset you need to:

   * `cd python_modules/roadsense/run/predict`
   * fill `predict_config.json` with appropriate values
   * `sh predict.sh` 
      
JSON Parameter Description:   
      
   * `bundle_path`: the folder created by `train.sh`
   * `pred_input_folder`: the drive folder path containing preprocessed data
   * `pred_output_folder`: path where the csv with the hazard detections should be generated

   
   