# Traffic signs object detection using [retinanet - Focal Loss for Dense Object Detection]
For more details, please refer to [arXiv paper](https://arxiv.org/abs/1708.02002).
Original Keras implementation from [fizyr](https://github.com/fizyr/keras-retinanet)

This code was tested with:

- CUDA_VERSION 9.0.176  (required to be installed on the machine)
- CUDNN_VERSION 7.0.5.15 (required to be installed on the machine)
- Tensorflow 1.8.0
- Keras 2.1.6
- Ubuntu 16.04.4


**A. Get the artifacts**

Download:
 Get the trips using trip downloader
    The trip downloader config file requers the next values on the following parameters:
    * signs:SIGNPOST_GENERIC
    * sign components flags: yes
    * min_signs_size: 3 (the size filtering will be done at the signpost level, as this filter also removes components)

Preprocess training data:
 1. Set the config preprocess_sign_components_data.json
    * input_meta_file_path: path to the initial meta containing generic signposts as proto rois
    * images_path: path to the full images
    * min_size: all the generic signposts having the biggest side less than min_size pixels won't be processed
    * min_nr_rois_inside_class: all the components classes with less than min_nr_rois_inside_class won't be added on the
        new generated meta file
    * replace_class_names: contains replacement classes for old tags that are not used at this point
 2. Run the preprocess_sign_components_data.py from scrips

**B. Installation**

We assume that everything that is going to run, will run from a docker container. Just execute:

1. build the docker image used for development:
    * `chmod +x ./docker_build_image_retinanet_signpost_components_detector.sh`
3. run the container: `sudo nvidia-docker run -v /data/:/data/ --net=host --name retinanet_signpost_components_detector_mq -d -it telenav/sign_components_detector_retinanet`

4. run to see the logs:
    * `sudo docker logs -f retinanet_signpost_components_detector_mq`

**C. Training**

1. Edit _train.sh_ script located in the current folder of the running docker container: `train.sh`
    * When one wants to train starting from:
        * imagenet weights just use as parameter _--imagenet-weights_.
        * from custom Keras weights just use for _WEIGHTS_ parameter a value like _/data/model/retinanet_resnet50_traffic_signs_v002.h5_ and for scripts the parameter _--weights $WEIGHTS_ instead of _--imagenet-weights_.
        * a previous snapshot just set the snapshot parameter to a value like _./snapshots/resnet50_traffic_signs_01.h5
    * For _TRAIN_PATH_ parameter set the value as _/data/train_data_.
    * For _VALIDATION_PATH_ parameter set the value  _/data/test_data_.
   For a more accurate evaluation during training phase you can split the train data in train and validation dataset.
2. Run `train.sh` script. The model checkpoints (weights) will be saved in _./snapshots_

**D. Predicting**

1. Edit _predict.sh_ script located in the current folder of the running docker container: `predict.sh`
    * For _WEIGHTS_ parameter:
        * set the path to the trained model file, like _/data/model/retinanet_resnet50_traffic_signs_v002.pb_. or
            to a given snapshot like  _./snapshots/resnet50_traffic_signs_01.h5
    * For _TRAIN_META_FILE_ set the value: _/data/train_data/rois.bin_
    * _INPUT_PATH_ is the folder containing the images you want to use for your detections. Set its value to your's validation dataset or to _/data/test_data_.
    * _THRESHOLD_FILE_ is a path to a json file containing confidence thresholds per class. The parameter's value can be set to:
        * _SAME_ : then all classes will have as minimum threshold the value specified in _LOWEST_SCORE_THRESHOLD_.
        * _/data/model/classes_thresholds.json_ : when one wants to use the thresholds already computed for a given model. The _classes_thresholds.json_ can be generated as is explained at point F.
    * For _LOWEST_SCORE_THRESHOLD_ see the description of _THRESHOLD_FILE_ above.
    * _OUTPUT_PATH_ should be_./snapshots/resnet50_traffic_signs_01.h5 set to a folder where the images with detected traffic signs will be saved (e.g _/data/output_). Also, in that folder will be generated the file _rois_retinanet.bin_ containing predicted ROIs for all images serialized in protobuf format.
2. Run `predict.sh` script.

**E. Basic evaluation**

1. Edit _evaluate.sh_ script located in the current folder of the running docker container: _evaluate.sh_
    * For _TEST_ROI_ parameter: set the path to the file containing ground truth ROIs for traffic signs of yours validation dataset (e.g. _/data/train_data/rois.bin_).
    * For _PREDICT_ROI_ parameter: set the path to the file containing serialized detections in protobuf format generated at step D in _OUTPUT_PATH_ folder (e.g. _./output/rois_retinanet.bin_).
    * _RESULT_FILE_ parameter indicates the text file where evaluations metrics will be saved.
2. Run `evaluate.sh` script.

**F. Optional: Generates best confidence thresholds maximizing the metric TP/(TP+FP+FN)**

1. Edit _generate_best_thresholds.sh_ script located in the current folder of the running docker container.
    * For _TEST_ROI_ parameter: set the path to the file containing ground truth ROIs for traffic signs in the validation dataset.
    * For _PREDICT_ROI_ parameter: set the path to the file containing serialized detections in protobuf format generated at step D in _OUTPUT_PATH_ folder (e.g. _./output/rois_retinanet.bin_).
    * _RESULT_FILE_ parameter indicates the path to the json file where best confidence thresholds will be saved. Later, this file can be used at step D for parameter _THRESHOLD_FILE_.
2. Run `generate_best_thresholds.sh` script.


Workflow e.g
1. Download the data set using
1. Create the new meta file and generic signpost's crops from the full pictures.
2. Split the data set into train/test.
3. Train a model using the train set and test set in order to evaluate.
4. Run a base predict with a low score threshold (e.g 0.1) on the train set.
5. Generate best thresholds using the base predict as predict roi and run a new predict using the resulted thresholds
6. Evaluate the model on the test set with the obtained thresholds

***Change log***
**1.0.0**
    * date: 03.09.2018
    * protobuf version: 1.0.0
    * keras-retinanet: 0.3.1 (`https://github.com/fizyr/keras-retinanet/releases/tag/0.3.1`)
    * docker:

**1.0.1**
    * date: 24.10.2018
    * protobuf version: 1.0.5
    * keras-retinanet: 0.3.1 (`https://github.com/fizyr/keras-retinanet/releases/tag/0.3.1`)

**1.1.0**
    * date: 11.12.2018
    * protobuf version: 1.0.6
    * keras-retinanet: 0.3.1 (`https://github.com/fizyr/keras-retinanet/releases/tag/0.3.1`)
    * refactoring for real sign components support in train, predict, evaluate
    
