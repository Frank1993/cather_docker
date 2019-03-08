import os
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
from keras.callbacks import ModelCheckpoint

import apollo_python_common.io_utils as io_utils
import roadsense.scripts.general.utils as utils
from roadsense.scripts.general.config import ConfigParams as cp, HazardType as HazardType
from roadsense.scripts.general.dataset_preprocessor import DatasetPreprocessor

class AbstractTrainer:
    def __init__(self, train_config):
        self.train_config = train_config
        self._augment_train_config()
        self.dataset_config = io_utils.json_load(self.train_config[cp.DATASET_CONFIG_PATH])
        self.class_2_index = self.train_config[cp.CLASS_2_INDEX]
        self.index_2_class = self.train_config[cp.INDEX_2_CLASS]

        self.preprocessor = DatasetPreprocessor(train_config)
        self.X_train, self.y_train_ohe, self.X_test, self.y_test_ohe, self.test_df = self.preprocessor.get_train_test_data()
        
        self.model = self._build_model()
        
    def _build_model(self):
        nr_classes = len(self.y_train_ohe[0])
        input_dim = self.X_train.shape[1]        
        return self.get_model(input_dim, nr_classes)

    def get_model(self, input_dim, nr_classes):
        pass
    
    def _get_index_dicts(self):
        index_2_class = dict(list(enumerate(self.train_config[cp.KEPT_HAZARDS] + [HazardType.CLEAR])))
        class_2_index = {v: k for k, v in index_2_class.items()}
        return index_2_class, class_2_index

    def _augment_train_config(self):
        index_2_class, class_2_index = self._get_index_dicts()
        self.train_config[cp.INDEX_2_CLASS], self.train_config[cp.CLASS_2_INDEX] = index_2_class, class_2_index

    def train_model(self):
        print("Training model...")

        best_ckpt_path = os.path.join(self.train_config[cp.CKPT_FOLDER], "model_temp.h5")
        self.model.fit(self.X_train, self.y_train_ohe,
                       validation_data=(self.X_test, self.y_test_ohe),
                       batch_size=self.train_config[cp.BATCH_SIZE],
                       epochs=self.train_config[cp.EPOCHS],
                       callbacks=[
                           ModelCheckpoint(best_ckpt_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')])

        self.model.load_weights(best_ckpt_path)

        return self.model

    def _save_model_to_json(self, json_path):
        model_json = self.model.to_json()
        with open(json_path, "w") as json_file:
            json_file.write(model_json)
    
    def _save_model(self, output_folder):
        self._save_model_to_json(os.path.join(output_folder, "model_structure.json"))
        self.model.save_weights(os.path.join(output_folder, "model_weights.h5"))

    def _save_config(self, output_folder):
        io_utils.json_dump(self.train_config, os.path.join(output_folder, "train_config.json"))

    def serialize_model_and_config(self):
        output_folder = self.train_config[cp.BUNDLE_PATH]
        io_utils.create_folder(output_folder)

        self._save_model(output_folder)
        self._save_config(output_folder)
        
        
    def predict_with_model(self, df):
        return self.model.predict(df, batch_size=self.train_config[cp.BATCH_SIZE], verbose=1)
    
    def _get_best_conf(self, conf_levels, f_scores):
        return conf_levels[np.argmax(f_scores)]

    def _keep_only_specific_hazard(self, pred_proba,index_of_hazard):
        modified_proba = []
        for prob_arr in pred_proba:
            modified_prob_arr = np.zeros(pred_proba.shape[1])
            modified_prob_arr[index_of_hazard] = prob_arr[index_of_hazard]
            modified_proba.append(modified_prob_arr)
            
        return np.asarray(modified_proba)
    
    def get_best_conf_threshold(self, y_pred_proba):
        print("Generating best conf threshold...")
        conf_levels = np.arange(0.01, 1, 0.05)

        y_test = [self.index_2_class[np.argmax(y)] for y in self.y_test_ohe]
        labels = list(self.class_2_index.keys())
        
        hazard_2_best_conf = {}
        for kept_hazard in reversed(self.train_config[cp.KEPT_HAZARDS]):
            print(f"Computing best thresholds for {kept_hazard}")
            hazard_index = self.class_2_index[kept_hazard]
            
            precisions, recalls, f_scores = [], [], []

            for conf_thresh in tqdm(conf_levels):
                specific_hazard_pred_proba = self._keep_only_specific_hazard(y_pred_proba,self.class_2_index[kept_hazard])
                y_pred = utils.keep_high_conf_hazards(specific_hazard_pred_proba, 
                                                      {kept_hazard:conf_thresh}, 
                                                      self.class_2_index)

                prec, recall, f_score, _ = precision_recall_fscore_support(y_test, y_pred, labels=labels)
                precisions.append(prec[hazard_index])
                recalls.append(recall[hazard_index])
                f_scores.append(f_score[hazard_index])

                print("{0:.3f} ===> P {1:.3f}".format(conf_thresh, prec[hazard_index]))
                print("{0:.3f} ===> R {1:.3f}".format(conf_thresh, recall[hazard_index]))
                print("{0:.3f} ===> F {1:.3f}".format(conf_thresh, f_score[hazard_index]))
                print("---")

                best_conf =  self._get_best_conf(conf_levels, f_scores)
                hazard_2_best_conf[kept_hazard] = best_conf

        return hazard_2_best_conf
