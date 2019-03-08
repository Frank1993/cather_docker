import pandas as pd
import numpy as np
from itertools import chain
from tqdm import tqdm
from sklearn.utils import shuffle
import os
import itertools
import random

class TextGenerator:
    CITY_KEY = "city"
    UPPERCASE_PROB = 0.3
    SLASH_PROB = 0.5

    def __init__(self, resources_path):
        self.csv_path = os.path.join(resources_path, "csv/north_america_pois.csv")

    def __get_allowed_letters(self):
        lower_letters = "q w e r t y u i o p a s d f g h j k l z x c v b n m - / . '".split(" ")
        upper_letters = [l.upper() for l in lower_letters]
        numbers = [str(i) for i in range(10)]

        return lower_letters + upper_letters + numbers + [" "]

    def __has_correct_chars(self, text, allowed_letters):
        for c in text:
            if c not in allowed_letters:
                return False
        return True

    def __read_texts_from_csv(self):
        return [str(text) for text in pd.read_csv(self.csv_path, names=[self.CITY_KEY])[self.CITY_KEY].tolist()]

    def __flatten(self, list_of_lists):
        return list(chain.from_iterable(list_of_lists))

    def __get_word_change_dict(self):
        return {
            "Road": "Rd",
            "Street": "St",
            "Avenue": "Ave",
            "Drive": "Dr",
            "Boulevard": "Blvd",
            "Freeway": "Fwy"
        }

    def __randomize_number(self,number):
        is_number = all([c.isdigit() for c in number])      
        if not is_number:
            return number
        
        if len(number) == 2 and np.random.random() < self.SLASH_PROB:
            number = f"{number[0]}/{number[1]}"
        elif len(number) == 3 and np.random.random() < self.SLASH_PROB:
            number = f"{number[0]} {number[1]}/{number[2]}"
            
        return str(number)
        
        
    def __randomize_text(self,text):
        return text.upper() if np.random.random() < self.UPPERCASE_PROB else text
    
    def __randomize(self, text):
            
        text = self.__randomize_text(text)
        text = self.__randomize_number(text)
        
        return text

    def __change_word(self, word, word_change_dict):
        return word_change_dict[word] if word in word_change_dict else word

    def __replace_words(self, text):
        word_change_dict = self.__get_word_change_dict()
        return " ".join([self.__change_word(word, word_change_dict) for word in text.split(" ")])

    def __with_suffix(self,texts, suffix):
        return [f"{t} {suffix}" for t in texts]

    def __generate_slash_miles(self, nr_texts):
        whole_miles = ['1','2','3']
        percentage_miles = ['1/2','1/4','3/4']
        cart_product = list(itertools.product(whole_miles,percentage_miles))

        combined_miles = [f"{w_m} {p_m}" for w_m,p_m in cart_product]

        texts = percentage_miles + combined_miles + \
                self.__with_suffix(percentage_miles, "mile") + \
                self.__with_suffix(combined_miles, "miles")

        final_texts = [random.choice(texts) for _ in range(nr_texts)]
        return final_texts


    def generate_texts(self, nr_texts_per_length, char_limit):

        allowed_letters = self.__get_allowed_letters()

        texts_from_csv = self.__read_texts_from_csv()
        texts_from_numbers = [str(i) for i in range(1000)]

        texts = self.__flatten([texts_from_csv, texts_from_numbers])

        texts = [text for text in tqdm(texts) if self.__has_correct_chars(text, allowed_letters)]
        texts = [self.__replace_words(text) for text in tqdm(texts)]
        words = self.__flatten([text.split(" ") for text in texts])
        texts += words
        texts = list(set(texts))
        texts = shuffle(texts, random_state=0)

        final_texts = []

        for text_length in tqdm(range(2, char_limit + 1)):

            random_texts = [text for text in texts if text_length == len(text)]
            random_texts = [self.__randomize(text) for text in random_texts]

            if len(random_texts) < nr_texts_per_length:
                scale_factor = (nr_texts_per_length // len(random_texts)) + 1

                all_random_texts = []
                for i in range(scale_factor):
                    all_random_texts += random_texts

                random_texts = all_random_texts

            final_texts += random_texts[:nr_texts_per_length]

        # add 10% extra texts containing "/"
        final_texts += self.__generate_slash_miles(len(final_texts)//10)

        final_texts = shuffle(final_texts, random_state=0)
        return final_texts
