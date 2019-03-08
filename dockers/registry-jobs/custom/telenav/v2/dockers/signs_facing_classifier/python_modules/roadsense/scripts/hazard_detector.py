import os

import apollo_python_common.io_utils as io_utils
import numpy as np
import roadsense.scripts.utils as utils
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from roadsense.scripts.config import ConfigParams as cp
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

tqdm.pandas()


class HazardDetector:

    def __init__(self, train_config):
        self.train_config = train_config
        self.dataset_config = io_utils.json_load(train_config[cp.DATASET_CONFIG_PATH])
        self.class_2_index = self.train_config[cp.CLASS_2_INDEX]
        self.index_2_class = self.train_config[cp.INDEX_2_CLASS]

    def get_model(self, input_dim, nr_classes):
        model = Sequential()
        model.add(Dense(1024, input_dim=input_dim, activation='relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.3))

        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.3))

        model.add(Dense(nr_classes, activation='softmax'))
        model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, X_train, y_train_ohe, X_test, y_test_ohe):
        print("Training model...")
        nr_classes = len(y_train_ohe[0])

        model = self.get_model(X_train.shape[1], nr_classes)

        best_ckpt_path = os.path.join(self.train_config[cp.CKPT_FOLDER], "model_temp.h5")
        model.fit(X_train, y_train_ohe,
                  validation_data=(X_test, y_test_ohe),
                  batch_size=self.train_config[cp.BATCH_SIZE],
                  epochs=self.train_config[cp.EPOCHS],
                  callbacks=[
                      ModelCheckpoint(best_ckpt_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')]
                  )

        model.load_weights(best_ckpt_path)

        return model

    def save_model(self, model, output_folder):
        self.save_model_to_json(model, os.path.join(output_folder, "model_structure.json"))
        model.save_weights(os.path.join(output_folder, "model_weights.h5"))

    def save_model_to_json(self, model, json_path):
        model_json = model.to_json()
        with open(json_path, "w") as json_file:
            json_file.write(model_json)

    def __get_best_conf(self, conf_levels, f_scores):
        return conf_levels[np.argmax(f_scores)]

    def get_best_conf_threshold(self, y_pred_proba, y_test_ohe):
        print("Generating best conf threshold...")
        conf_levels = np.arange(0.01, 1, 0.01)

        y_test = [self.index_2_class[np.argmax(y)] for y in y_test_ohe]
        labels = list(self.class_2_index.keys())
        kept_hazard = self.train_config[cp.KEPT_HAZARDS][0]  # todo add support for multiple hazards
        hazard_index = self.class_2_index[kept_hazard]

        precisions, recalls, f_scores = [], [], []

        for conf_thresh in tqdm(conf_levels):
            y_pred = utils.keep_high_conf_hazards(y_pred_proba, conf_thresh, self.class_2_index)

            prec, recall, f_score, _ = precision_recall_fscore_support(y_test, y_pred, labels=labels)
            precisions.append(prec[hazard_index])
            recalls.append(recall[hazard_index])
            f_scores.append(f_score[hazard_index])

            print("{0:.3f} ===> {1:.3f}".format(conf_thresh, f_score[hazard_index]))

        return self.__get_best_conf(conf_levels, f_scores)

    def predict_with_model(self, model, X_test):
        return model.predict(X_test, batch_size=self.train_config[cp.BATCH_SIZE], verbose=1)
