import os
from apollo_python_common.lightweight_types import AttributeDict

INVALID_IMG_PRED = "-1"
IMAGE_RATIO = 16 / 9

AVAILABLE_ALGORITHMS = ["image_quality", "image_orientation", "roi_classifier", "signs_facing_classifier"]


class PipelineParamsBuilder:

    @staticmethod
    def build_params(base_path,
                     nr_images_per_class,
                     suffix,
                     with_vp_crop,
                     img_dim=336,
                     square_image = False,
                     keep_aspect=False
                     ):
        '''
        :param base_path: base path for the work directory
        :param nr_images_per_class: how many images for each class to take into consideration
        :param suffix: unique identifier for the generated dataset
        :param with_vp_crop: determines if the image is cut at Vanishing Point or not
        :param img_dim: the dimension at which to resize the images
        :param keep_aspect: determines if the image is resized keeping its aspect ratio or not
        '''

        # Params

        params = AttributeDict()

        params.nr_images_per_class = nr_images_per_class
        params.suffix = suffix
        params.with_vp_crop = with_vp_crop
        params.keep_aspect = keep_aspect

        # Derived Params
        params.base_img_path = os.path.join(base_path,"raw_imgs")
        params.processed_base_path = os.path.join(base_path,"processed")

        params.classes = sorted(os.listdir(params.base_img_path))

        params.class_2_classIndex = dict(zip(params.classes,
                                             [params.classes.index(current_class) for current_class in params.classes]))
        params.classIndex_2_class = {v: k for k, v in params.class_2_classIndex.items()}

        params.nr_entries_per_split = 256
        params.train_percentage = 0.8
        
        if square_image:
            params.img_size = (img_dim, img_dim)            
        else:
            params.img_size = (int(img_dim * IMAGE_RATIO), img_dim)

        # Paths
        params.working_dir = "classes={}_imgs={}_size={}_{}/".format(len(params.classes),
                                                                     params.nr_images_per_class,
                                                                     img_dim,
                                                                     params.suffix)

        params.processed_path = os.path.join(params.processed_base_path,params.working_dir)

        params.train_path = os.path.join(params.processed_path,"train")
        params.test_path = os.path.join(params.processed_path,"test")
        params.params_path = os.path.join(params.processed_path,"params")

        params.train_df_path = os.path.join(params.train_path,"df")
        params.test_df_path = os.path.join(params.test_path,"df")

        params.train_img_path = os.path.join(params.train_path,"imgs")
        params.test_img_path = os.path.join(params.test_path,"imgs")

        params.conv_layer_name = "mixed6"

        params.train_conv_path = os.path.join(params.train_path,params.conv_layer_name)
        params.test_conv_path = os.path.join(params.test_path,params.conv_layer_name)

        return params


class Column:
    SEQ_ID_COL = 'seq_id'
    LABEL_CLASS_COL = 'label_class'
    IMG_NAME_COL = 'img_name'
    WAY_ID_COL = 'way_id'
    LABEL_COL = 'label'
    IMG_COL = 'img'
    HEIGHT_RATIO_COL = 'height_ratio'
    HEIGHT_BEFORE_RESIZE_COL = 'height_before_resize'
    VALID_HEIGHT_RATIO_COL = 'valid_height_ratio'
    FULL_IMG_NAME_COL = 'full_img_name'
    CONV_IMG_COL = 'conv_img'
    PRED_COL = 'pred'
    PRED_CLASS_COL = 'pred_class'
    CORRECT_COL = 'correct'
    PRED_CONF_COL = 'pred_conf'
    NR_SEQS_COL = 'nr_seqs'


class NetworkCfgParams:
    NETWORK_CONFIG_PARAM = "network_config"
    NR_CONV_BLOCKS_PARAM = "nr_conv_blocks"
    NR_FILTERS_PARAM = "nr_filters"
    DROPOUT_LEVEL_PARAM = "dropout_level"
    TRAIN_MODEL_PARAMS_PATH = "train_model_params_path"
    TEST_MODEL_PARAMS_PATH = "test_model_params_path"
    NR_EPOCHS_PARAM = "nr_epochs"
    TRAIN_PROGRESS_CSV_PARAM = "train_progress_csv"
    FTP_OUTPUT_PARAM = "ftp_output_path"
    WEIGHTS_LOCAL_OUTPUT_PATH_PARAM = "weights_local_output_path"
    WITH_DIFFERENT_TEST_SET_PARAM = "with_different_test_set"
    
class PredictorCfgParams:
    FTP_BUNDLE_PATH_PARAM = "ftp_bundle_path"
    NR_IMGS_PARAM = "nr_imgs"
    OUTPUT_FOLDER_PARAM = "output_folder"
    INPUT_FOLDER_PARAM = "input_folder"