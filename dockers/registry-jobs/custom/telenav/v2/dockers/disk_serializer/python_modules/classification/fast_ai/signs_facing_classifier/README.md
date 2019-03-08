# Signs Facing Classifier 2.0 

**General Info**

This module is responsible for classifying the facing of detected traffic signs. The classification is made into three categories:
front, left or right. Front is for when the traffic sign faces the driver, left if for when the driver has to turn left to face
the sign and right is for when the driver has to turn right to face the sign.  


**A. Dataset Description**

The dataset is constructed by tagging ROIs in our detector train set as front, left or right. The procedure is as follows:
  * a subset of images from our detector train sets were selected for tagging
  * we tagged the images using quick_tag_crop which resulted in folders with front, left and right samples
  * we have done this in 2 separate rounds, one for version 1.0 and another for version 2.0
  * the tagged images from both rounds were merged together to create one large set of images for pre-processing.
    
**B. Dataset Preprocessing**

1. ROI matching.

In order to run the model training, we have to preprocess the data to match the ROIs with the ROIs from the object detection component
This is done in the dataset_matcher.py script. It takes all the images in the merged ROIs folder, creates a dataframe, loads the
corresponding object detector rois.bin files and matches the tagged ROIs with the detected ROIs using IOU as a measurement.
The matched ROIs will then be updated with the ground truth (object detector) roi coordinates and re-cropped accordingly.
Another step is to split the data into train and test datasets. This is done using a stratified 70/30 split over the
front, left and right classes.
   
   
JSON Parameter Description:

   * `full_set_dir`: the directory which contains all tagged merged data, before preprocessing
   * `new_orig_img_dir`: the directory with the 09_2018 object detector train set - the source images
   * `old_orig_img_dir`: the directory with the 04_2018 object detector train set - the source images
   * `new_detections_file`: the rois.bin containing the object detector results for the 09_2018 train set
   * `old_detections_file`: the rois.bin containing the object detector results for the 04_2018 train set
   * `train_set_dir`: the train set directory where we output the matched ROIs after the split
   * `test_set_dir`: the test set directory where we output the matched ROIs after the split
   * `matched_results_file`: the file where we save the matched results, as an intermediate step
   * `split_percentage`: the train/test split percentage
   * `iou_threshold`: the IOU threshold used for matching
   
To run:
   * `cd python_modules/classification/fast_ai/signs_facing_classifier/run/dataset`
   * fill `match_dataset_cfg.json` with appropriate parameters
   * `sh match_dataset.sh`

2. ROI cropping.
Another processing step is to enhance the cropped ROIs by adding some context around the detected traffic sign
and making the resulting crop square. This is done by using a crop factor to add a percentage of context, then
cutting the image such that the resulting rectangle is a square. 


JSON Parameter Description:

   * `train_input_dir`: the train set directory that resulted after the match and split
   * `train_output_dir`: the train set output directory where the square crops are placed
   * `test_input_dir`: the test set directory that resulted after the match and split
   * `test_output_dir`: the test set output directory where the square crops are placed
   * `sq_crop_factor`: the square crop factor - percentage of context added
   * `new_orig_img_dir`: the directory with the 09_2018 object detector train set - the source images
   * `old_orig_img_dir`: the directory with the 04_2018 object detector train set - the source images
   
To run:
   * `cd python_modules/classification/fast_ai/signs_facing_classifier/run/dataset`
   * fill `roi_cropper_cfg.json` with appropriate parameters
   * `sh roi_cropper.sh`
   
**C. Model Training**

In order to run the model training over the preprocessed dataset you need to:

   * `cd python_modules/classification/fast_ai/signs_facing_classifier/run/model`
   * fill `train_cfg.json` with appropriate parameters
   * `sh train.sh` 
   
   
JSON Parameter Description:

   * `imgs_dir`: the directory containing preprocessed train images sorted into front, left and right folders
   * `frozen_backbone_model`: name of the model to be saved in the frozen backbone stage
   * `final_model`: name of the final model to be saved after unfreezing the backbone
   * `label_list_file`: file into which we save the order of the labels internally used by fast.ai
   * `epochs`: number of epochs to run training with frozen backbone
   * `unfreeze_epochs`: number of epochs to run training with unfrozen backbone
   * `frozen_lr`: the learning rate for training with the frozen backbone
   * `unfreeze_lr1`: the low end learning rate interval for training after unfreezing the backbone
   * `unfreeze_lr2`: the high end learning rate interval for training after unfreezing the backbone
   * `pct_start`: the learning rate increase percentage start
   * `backbone_model`: the backbone model used for pretrained weights and the underlying network architecture
   * `batch_size`: the training cycle batch size
   * `image_size`: the size to which the images will be resized by the library before passing them in the training model
   * `tfms_max_rotate`: transforms applied by fast.ai - max rotation angle for images
   * `tfms_max_warp`: transforms applied by fast.ai - max warp applied to images
   * `tfms_flip_vert`: transforms applied by fast.ai - flip images vertically - true or false
   * `tfms_do_flip`: transforms applied by fast.ai - flip images horizontally - true or false


**D. Prediction**

In order to run the prediction over the preprocessed dataset you need to:

   * `cd python_modules/classification/fast_ai/signs_facing_classifier/run/model`
   * fill `predict_cfg.json` with appropriate parameters
   * `sh predict.sh` 
      
JSON Parameter Description:   
      
   * `imgs_dir`: the directory containing preprocessed predict images sorted into front, left and right folders
   * `model_dir`: the directory used by the fast.ai Learner as an empty data bunch loader - this must contain the trained model under `models`
   * `model_name`: the name of the trained model used for inference
   * `label_list_file`: file into which we save the order of the labels internally used by fast.ai
   * `backbone_model`: the backbone model used for pretrained weights and the underlying network architecture
   * `batch_size`: the training cycle batch size
   * `image_size`: the size to which the images will be resized by the library before passing them in the training model
   * `tfms_max_rotate`: transforms applied by fast.ai - max rotation angle for images
   * `tfms_max_warp`: transforms applied by fast.ai - max warp applied to images
   * `tfms_flip_vert`: transforms applied by fast.ai - flip images vertically - true or false
   * `tfms_do_flip`: transforms applied by fast.ai - flip images horizontally - true or false
   * `tta`: do test time augmentation - true or false
   
   