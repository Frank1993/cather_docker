import os
import shutil
import unittest
from glob import glob

import pandas as pd
from ocr.scripts.prediction.folder_ocr_predictor import FolderOCRPredictor
from ocr.scripts.prediction.ocr_predictor import ResizeType
from ocr.scripts.text_correction.signpost_text_corrector import SignpostTextCorrector
from unittests.utils import resource_utils
from unittests.utils.resource_utils import LOCAL_TEST_RESOURCES_FOLDER


class OCRPerformanceTest(unittest.TestCase):
    CORRECT = "correct"
    ERROR = "error"
    GT_CHAR_LENGTH = "gt_char_length"

    ftp_imgs_path = "/ORBB/data/test/python/ocr/corrected_test_set_v2.zip"
    local_imgs_folder = "test/"

    ftp_ckpt_path = "/ORBB/data/test/python/ocr/ocr_ckpt.zip"
    local_ckpt_folder = "ckpt/"

    EXPECTED_ACCURACY = 0.9267
    EXPECTED_AVG_CHAR_ERROR = 0.0164

    def setUp(self):
        print("Downloading resources from FTP...")
        resource_utils.ensure_test_resource(self.ftp_imgs_path, self.local_imgs_folder)
        resource_utils.ensure_test_resource(self.ftp_ckpt_path, self.local_ckpt_folder)

    def tearDown(self):
        print("Cleaning up imgs resources...")
        if os.path.exists(LOCAL_TEST_RESOURCES_FOLDER):
            shutil.rmtree(LOCAL_TEST_RESOURCES_FOLDER)

    def test_performance(self):
        full_imgs_path = os.path.join(LOCAL_TEST_RESOURCES_FOLDER, self.local_imgs_folder, "corrected_test_set_v2")
        full_ckpt_path = os.path.join(LOCAL_TEST_RESOURCES_FOLDER, self.local_ckpt_folder, "model.ckpt-71251")

        predict_folders = glob(full_imgs_path + "/*")

        spell_checker_resources_path = None
        text_corrector = SignpostTextCorrector(spell_checker_resources_path)

        predictor = FolderOCRPredictor("fake_dataset", full_ckpt_path, ResizeType.DEFORM, text_corrector)

        pred_df_list = [predictor.predict_on_folder(folder, with_gt=True, min_component_size=25)
                        for folder in predict_folders]
        pred_df = pd.concat(pred_df_list)

        accuracy = round(pred_df[self.CORRECT].mean(), 4)
        average_char_error = round(pred_df[self.ERROR].sum() / pred_df[self.GT_CHAR_LENGTH].sum(), 4)

        print(f"Accuracy {accuracy}")
        print(f"Average char eror {average_char_error}")
        assert (accuracy == self.EXPECTED_ACCURACY)
        assert (average_char_error == self.EXPECTED_AVG_CHAR_ERROR)
