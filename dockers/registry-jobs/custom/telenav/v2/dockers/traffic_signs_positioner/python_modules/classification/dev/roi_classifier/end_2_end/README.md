# Roi Classifier End 2 End  


These are the instructions for running the end 2 end train & predict of the ROI Classifier.
 
####1. Navigate to folder
*   Run `cd python_modules/classification/dev/end_2_end`
      
####2. Modify `config.json`
*   `train_preds_file`: train set predictions file which was created by the traffic signs detector 
    *   `"train_preds_file": "/home/docker/apollo/datasets/roi_classifier_09_2018/flaviu_preds/rois_flaviu_train_v2.bin"`
*   `train_gt_folder`: train set ground truth folder
    *   `"train_gt_folder": "/data_4gpu/traffic_signs_09_2018/train_new"`
*   `test_preds_file`: test set predictions file which was created by the traffic signs detector
    *   `"test_preds_file: "/home/docker/apollo/datasets/roi_classifier_09_2018/flaviu_preds/rois_flaviu_test_v2.bin"`
*   `test_gt_folder`: test set ground truth folder
    *   `"test_gt_folder": "/data_4gpu/traffic_signs_09_2018/test"`
*   `output_path`: folder path where all the results will be put
    *   `"output_path": "/home/docker/apollo/datasets/roi_classifier_09_2018/end_2_end_workspace_2"`
*   `ftp_output_path`: FTP path where to save the trained weights from the ROI Classifier
    *   `"ftp_output_path": "/ORBB/data/temp/test_bundle.zip"`
*   `classif_img_dim`: img size at which to train the classifier
    *   `"classif_img_dim":139`
*   `nr_epochs`: nr of epochs to train the classifier for
    *   `"nr_epochs":20`
*   `min_size`: min roi size for roi evaluation
    *   `"min_size": 25`
*   `iou_threshold`: iou threshold for roi evaluation
    *   `"iou_threshold": 0.25`
*   `dataset_classes_file`: the file containing the list of roi classes from which to create the classification dataset
    *   `"dataset_classes_file":"/home/docker/apollo/datasets/roi_classifier_09_2018/retinanet_preds/most_classes.json"`
*   `evaluate_classes_file`: the file containing the list of roi classes on which to make the final roi evaluation
    *   `"evaluate_classes_file": "/home/docker/apollo/datasets/roi_classifier_09_2018/retinanet_preds/selected_classes.json"` 
    
####3. Run script
*   Run `sh run_end_2_end.sh`
    
####4. Output Interpretation

In the `output_path` folder, the following folder structure and files will be created

*   `best_thresholds/`
    *   `best_thresholds.json`
        * optimal roi type confidences for the joint roi predictions (detector + classifier)        
*   `dataset/`
    *   `train_dataset/`, `test_dataset/`, `train_temp_dataset/`
        *   these are the 3 datasets that will be used for classification network train. The difference between `train_dataset` and `train_temp_dataset` is that the `train_dataset` contains rois extracted with various border sizes and also augmented whereas `train_temp_dataset` is only used to make predictions on the train set.
        
*   `network/`
    *   `weights/weights.h5`
        *   the best weights for the roi classifier
        
*   `postprocess/`
    *   `train/postprocessed_preds_0.5/rois.bin`: postprocessed train predictions (the initial prediction rois after classification filtering). The default confidence for classification has been used, namely 0.5
    *   `test/postprocessed_preds_0.X/rois.bin`: postprocessed test predictions (the initial prediction rois after classification filtering). Results using various confidences have been created
    
*   `predictions/`
    *   `train_preds.pkl`, `test_preds.pkl`: train/test predictions used internally by the script.
    
*   `results/`
    *   `best_results.txt`: path of the test roi file containing the highest test accuracy.
    *   `statistics.json`: roi evaluation on the test roi file containing the highest test accuracy.