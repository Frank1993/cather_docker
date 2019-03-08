# Signpost OCR 

**General Info**

This network performs OCR on the text components extracted from signpost. 
Original Code: https://github.com/tensorflow/models/tree/master/research/attention_ocr

It has been trained using a generated fake dataset of various signposts with text artificially written on them

**A. Generate Fake Dataset**
 
 The fake dataset will contain both the raw images, and the TF Records file needed to train the network.
 You can find the text written in every image in the generated CSV  
 
1. Navigate to target folder 
    * Run `cd python_modules/ocr/run/fake_dataset/`
      
2. Modify `fake_dataset_config.json`
    * Parameter Descriptions:
 
        * `resources_path`: local path of the resources folder. This folder contains the fonts, backgrounds and texts which will be used to generate the images. It can be donwloaded from FTP from `/ORBB/data/ocr/fake_dataset_resources`  
            * `"resources_path":"/home/docker/apollo/datasets/ocr_sign_posts/fake_dataset_resources"`   
        * `save_path`: The output folder where the dataset will be generated 
            * `"save_path": "/home/docker/apollo/datasets/ocr_sign_posts/fake_generated/few_images"`
        * `width`: The width of the generated images
            * `"width": 200`
        * `height`: The heightof the generated images
            * `"height": 50`
        * `nr_texts_per_length`: How many texts per length to generate. This will generate X texts of length 2, X texts of length 3, etc. 
            * `"nr_texts_per_length": 200`
        * `char_limit`: The maximum char length of texts. 
            * `"char_limit":16`
           
2. Run script
    * Run `sh create_fake_dataset.sh` 


**B. Train**
 
 The train config is located in `python_modules/ocr/attention_ocr/datasets`. The original TF documentation says that every dataset should contain it's own separate .py file. To train with other parameters, you can change the existing `fake_dataset.py` or create another file. In that case, you also need to import it in the `datasasets`' folder `__init__.py` file
   
1. Navigate to train config folder 
    * Run `cd python_modules/ocr/attention_ocr/datasets/fake_dataset.py`
    * Parameter Descriptions: 
        *  `DEFAULT_DATASET_DIR`: put the path to the `tf_data` folder in the generated fake dataset
            *  `DEFAULT_DATASET_DIR = "/home/docker/apollo/datasets/ocr_sign_posts/fake_generated/2mil_images_v3/tf_data"`
        *  `DEFAULT_CONFIG`:
            * `name`: name of dataset. little importance
                * `'name': 'Fake Dataset'`
            * `splits`: name of the folders inside `DEFAULT_DATASET_DIR`. In our case, `train` and `test`.
                * `size`: number of images
                * `pattern`: the naming pattern of the TF records starting from  `DEFAULT_DATASET_DIR`.
                
                *`'splits': {  
                    'train': {
                        'size': 2000000,
                        'pattern': 'train/train*'
                    },
                    'test': {
                        'size': 50000,
                        'pattern': 'test/test*'
                        }
                        },`   
            * `charset_file_name`: Name of the charset file containing the mappings form char to index. Should exists in `DEFAULT_DATASET_DIR`. It must be put manually there. You can also find it in the resources folder
                * `'charset_filename': 'charset_size=134.txt'`
            * `image_shape`: Triplet containing the shape of the image (H,W,NR_CHANNELS)
                * `'image_shape': (50, 200, 3)`
            * `num_of_views`: This network can take as input multiple views of the same sign stitched together. We are not using that functionality and only provide 1 view of a sign
                * `'num_of_views': 1`
            * `max_sequence_length`: Max length for texts 
                * `'max_sequence_length': 16`
            * `null_code`: Code of the null character. Should be taken frm the `charset_file`
                * `null_code': 99`
        
2. Run script
    * Run `sh train.sh` 

**C. Predict**
 
1. Navigate to folder
    * Run `cd python_modules/ocr/run/predict/`
    
2. Modify `predit_config.json`
    * Parameter Descriptions:
        * `ckpt_path`: Path to saved checkpoints fom model training  
            * `"ckpt_path":"/home/docker/apollo/datasets/ocr_sign_posts/weights/weights_dataset=1mil_images_all_us_margins/model.ckpt-230975"`   
        * `dataset_name`: The name of the .py file which was used during training
            * `"dataset_name":"fake_dataset"`   
        * `predict_folder`: The folder containing the images which should be predicted 
            * `"predict_folder": "/home/docker/apollo/datasets/ocr_sign_posts/real_images_test_dataset"`   
        * `output_csv_path`: Path where to save the text predictions for each file
            * `"output_csv_path": "./ocr_predictions.csv"`   
        * `nr_imgs`: How many images to run the prediction on. If you want all, put -1
            * `"nr_imgs":-1`   
        * `with_evaluate`: For simple predicting purposes, this should be set ot false
            * `"with_evaluate": false`   
        * `min_component_size`: The minimum component size on which to run the prediction
            * `"min_component_size":25`   
        * `conf_thresh`: The minumum confidence for a prediction in order to be considered valid
            * `"conf_thresh":0.5`   
        

2. Run predict script
    * Run `sh predict.sh` 
  
**D. Evaluate**
 
1. Navigate to folder
    * Run `cd python_modules/ocr/run/evaluate/`
    
2. Modify `evaluate_config.json`
    * Parameter Descriptions:
        * `ckpt_path`: Path to saved checkpoints fom model training  
            * `"ckpt_path":"/home/docker/apollo/datasets/ocr_sign_posts/weights/weights_dataset=1mil_images_all_us_margins/model.ckpt-230975"`   
        * `dataset_name`: The name of the .py file which was used during training
            * `"dataset_name":"fake_dataset"`   
        * `predict_folder`: The folder containing the images which should be predicted. For evaluating, you should use the dataset in which the text in encoded in the name of the file 
            * `"predict_folder": "/home/docker/apollo/datasets/ocr_sign_posts/real_images_test_dataset"`   
        * `output_csv_path`: Path where to save the text predictions for each file
            * `"output_csv_path": "./ocr_predictions.csv"`   
        * `nr_imgs`: How many images to run the prediction on. If you want all, put -1
            * `"nr_imgs":-1`   
        * `with_evaluate`: For evaluating, this should be set to true
            * `"with_evaluate": true`   
        * `min_component_size`: The minimum component size on which to run the prediction
            * `"min_component_size":25`   
        * `conf_thresh`: The minumum confidence for a prediction in order to be considered valid
            * `"conf_thresh":0.5`   

2. Run predict script
    * Run `sh evaluate.sh` 
  
 
