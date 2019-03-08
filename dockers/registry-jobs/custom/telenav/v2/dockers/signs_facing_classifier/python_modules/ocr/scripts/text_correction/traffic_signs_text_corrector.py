import pandas as pd
from ocr.scripts.text_correction.abstract_text_corrector import AbstractTextCorrector
import Levenshtein


class TrafficSignsTextCorrector(AbstractTextCorrector):
    TEXT_COL = "text"
    NEW_LINE_PROXY_CHAR = "€"
    NEW_LINE_REPLACE_CHAR = " "

    def __init__(self, texts_path):
        self.possible_texts = pd.read_csv(texts_path, names=[self.TEXT_COL])[self.TEXT_COL].values

    def correct_text(self, pred_text):
        pred_text = pred_text.replace(self.NEW_LINE_PROXY_CHAR, self.NEW_LINE_REPLACE_CHAR)
        pred_2_distance_list = [(possible_text, Levenshtein.distance(pred_text, possible_text))
                                for possible_text in self.possible_texts]
        return sorted(pred_2_distance_list, key=lambda t: t[1])[0][0]
