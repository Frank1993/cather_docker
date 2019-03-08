# Classification Network 

**General Info**

These are the instructions for using the classification network. We currently have 3 components for which is can be run:
1.  image_orientation
2.  image_quality
3.  roi_classification

**A. Setup Environment**

1. Create docker image
    * Run `cd python_modules/classification/dev/docker/Dockerfile`
    * Run `sh build.sh`
 
2. Start Container
    * Run `sudo nvidia-docker run -d --name class -it -p 8887:8888 -v {LOCAL_PATH}:/home/docker/apollo/ telenav/class`

Every command for dataset building/train/predict must be run from inside the docker container.  

**B. Dataset Build**

1. Choose component
    * Run `cd python_modules/classification/dev/{COMPONENT_NAME}/dataset`
        * For example `cd python_modules/classification/dev/image_orientation/dataset`
2. Modify `dataset_config.json`
    * `base_path`: path to the folder which will be the workspace for that classification dataset. In that folder, you will need to have a `raw_imgs` folder containing the images separated in different folders depending on their class. For example:
        * `"base_path":"/images/animals"`
        * Assuming you have the following directory structure`/images/animals/raw_imgs/dogs` and `/images/animals/raw_imgs/cats`. 
    * `img_dim`: resize dimension. For example:
        * `"img_dim":336`
    * `nr_images_per_class`: how many images to include from each class. If you want all of them, just set a very high number here. For example:
        * `"nr_images_per_class":10000`
    * `augment_strategies`: strategies which will used to augment the dataset. This parameter must contain a dictionary with the following 3 key-value pairs: `flip`, `rotate_cw`, `rotate_ccw`. Each key represents an augment strategy, and the value is a boolean stating if that strategy should be used. Note: They will be applied individually on the dataset and not combined. For example:
        * `"augment_strategies" : {
            "flip": false,
            "rotate_cw": true,
            "rotate_ccw": true
            }`
            
     * `suffix`: simple string which will be appended to the folder name containing the processed dataset. It's purpose is to differentiate between multiple datasets constructed with different parameters. For example:
        * `"suffix":"all_data"`
     * `with_vp_crop`: boolean parameter which defines whether or not the image should be cut at the horizon line. For example:
        * `with_vp_crop:false`
     * `split_train_test_by_seq_id`: boolean parameter which defines the train-test split strategy. Regardless of the strategy, 80% of the data will be used for train and the rest for test. If set to false, the split will be performed randomly. If set to true, the split will be done according the the sequence_id, meaning that images from a certain sequence can only appear in the training set or the test set, but not in both. In order do this, the sequence_id must be embedded in the image name. Meaning an image with the original name `116287_1d629_58e702d684f7a.jpg` must be prechanged to `116287_SEQID_1d629_58e702d684f7a.jpg`. For example:
        * `"split_train_test_by_seq_id":false`
     * `square_image`: boolean parameter which determines the shape at which the images will be resized. If set to true, they will reshaped to a square, having both sides equal to `img_dim` parameter. Otherwise, they will be reshaped to a rectangle, having the small side equal to `img_dim` and the big side equal to 16/9 * `img_dim`            

2. Run create dataset script
    * Run `sh create_dataset.sh`
   
**C. Train**
 
1. Choose component
    * Run `cd python_modules/classification/dev/{COMPONENT_NAME}/train`
        * For example cd python_modules/classification/dev/image_orientation/train
    
2. Modify `train_config.json`
    * Parameter Descriptions:
        * `model_params_path`: path to the `model_params.json` file which contains all the information about the dataset. For example
            * assuming your `base_path` was `/images/animals`, you can find the file in `/images/animals/processed/{PROCESSED_DATASET_NAME}/params` folder. As a result, a valid value would be:
            * `"model_params_path": /images/animals/processed/{PROCESSED_DATASET_NAME}/params/model_params.json`   
        * `train_progress_csv`: path to a csv file in which training progress will we written, such as the train accuracy/loss and the validation accuracy/loss. For example:
            * `"train_progress_csv":"./progress.csv"`
        * `ftp_output_path`: ftp path where the trained model & weights will be saved. For example
            * `"ftp_output_path":"/ORBB/data/temp/test_bundle.zip"`
        * `network_config`: this parameter defines the architecture of the network which will be "put" on top of the base pretrained InceptionV3 network. It needs to be a dictionary containing the following keys `nr_conv_blocks`,`nr_filters`,`dense_size`,`dropout_level`. Fow example:
            * `"network_config": {
                    "nr_conv_blocks": 3,
                    "nr_filters": 32,
                    "dense_size": 128,
                    "dropout_level": 0.3
                }`
        
2. Run train script
    * Run `sh train.sh` 

**C. Predict**
 
1. Choose component
    * Run `cd python_modules/classification/dev/{COMPONENT_NAME}/predict`
        * For example cd python_modules/classification/dev/image_orientation/predict
    
2. Modify `predict_config.json`
    * Parameter Descriptions:
        * `ftp_bundle_path`:ftp path to the zip bundle containing the trained model. For example:
            *`"ftp_bundle_path":"/ORBB/data/temp/test_bundle.zip"`
        * `input_folder`: folder containing the images that should be predicted. For example:
            * `"input_folder":"/home/images/"`
        * `nr_imgs`: how many images should be predicted from the folder. If you want all, just set a really high number. For example
            * `"nr_imgs":100`
        * `output_folder`: output folder where the proto should be saved. For example
            * `"output_folder":"./output"`

2. Run predict script
    * Run `sh predict.sh` 
   