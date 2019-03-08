from apollo_python_common.ml_pipeline.config_api import MQ_Param


class OCR_MQ_Param(MQ_Param):
    DATASET = "dataset"
    CKPT_PATH = "ckpt_path"
    SPELL_CHECKER_PATH = "spell_checker_resources_path"
    MIN_COMPONENT_SIZE = "min_component_size"
    CONF_THRESH = "conf_thresh"
    TEXT_CORRECTION_RESOURCES_PATH = "text_correction_resources_path"
    ROI_CLASSES_TO_PREDICT_PATH = "roi_classes_to_predict_path"
    COMPONENTS = "components"
    SIGNPOST_COMP = "signpost_components"
    ROIS = "rois"
