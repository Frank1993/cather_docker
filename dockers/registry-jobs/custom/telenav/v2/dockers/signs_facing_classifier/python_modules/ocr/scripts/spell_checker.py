import pandas as pd
from tqdm import tqdm

tqdm.pandas()
from collections import Counter, defaultdict
from itertools import chain
import Levenshtein


class SpellChecker:
    ID_COL = "id"
    POI_COL = "poi"
    MAX_DISTANCE = 1

    def __init__(self, texts_source_path):
        texts, self.word_2_count = self.__read_source_data(texts_source_path)
        self.ngram_2_word_list = self.__get_ngram_2_word_list(texts)

    def __has_correct_chars(self, text, allowed_letters):
        for c in text:
            if c not in allowed_letters:
                return False
        return True

    def __flatmap(self, list_of_lists):
        return [item for list_ in list_of_lists for item in list_]

    def __flatten(self, listOfLists):
        return list(chain.from_iterable(listOfLists))

    def __get_ngrams(self, b, n=3):
        return [b[i:i + n] for i in range(len(b) - n + 1)]

    def __get_ngram_2_word_list(self, texts):
        text_2_ngram_list = [(text, self.__get_ngrams(text)) for text in tqdm(texts)]

        ngram_2_word_list = defaultdict(list)

        for text, ngram_list in tqdm(text_2_ngram_list):
            for ngram in ngram_list:
                ngram_2_word_list[ngram].append(text)

        return ngram_2_word_list

    def __replace_words(self, text):
        return " ".join([self.__change_word(word) for word in text.split(" ")])

    def __change_word(sels, word):
        word_change_dict = {
            "Road": "Rd",
            "Street": "St",
            "Avenue": "Ave",
            "Drive": "Dr",
            "Boulevard": "Blvd",
            "Freeway": "Fwy"
        }

        if word in word_change_dict:
            return word_change_dict[word]

        return word

    def __read_source_data(self, texts_source_path):
        allowed_letters = self.__get_allowed_letters()
        data_df = pd.read_csv(texts_source_path, nrows=None, names=[self.POI_COL])

        texts = [str(text).lower() for text in data_df[self.POI_COL].tolist()]
        texts = [text for text in tqdm(texts) if self.__has_correct_chars(text, allowed_letters)]
        texts = [self.__replace_words(text) for text in tqdm(texts)]
        words = self.__flatten([text.split(" ") for text in texts])
        words += ["exit", "exits", "freeway", "freeways"]

        word_2_count = Counter(words)

        words = list(set(words))

        return words, word_2_count

    def __get_allowed_letters(self):
        lower_letters = "q w e r t y u i o p a s d f g h j k l z x c v b n m - / . '".split(" ")
        upper_letters = [l.upper() for l in lower_letters]
        numbers = [str(i) for i in range(10)]

        letters = lower_letters + upper_letters + numbers + [" "]
        return letters

    def __get_words_with_matching_ngrams(self, target_word):

        target_ngrams = self.__get_ngrams(target_word)

        words_with_matching_ngrams = [self.ngram_2_word_list[ngram] for ngram in target_ngrams]

        return list(set(self.__flatmap(words_with_matching_ngrams)))

    def __get_close_matches(self, target_word, word_list, max_distance=10):
        word_2_distance_list = [(word, Levenshtein.distance(word, target_word)) for word in word_list]
        word_2_distance_list = [(w, d) for w, d in word_2_distance_list if d <= max_distance]
        word_2_distance_list = sorted(word_2_distance_list, key=lambda t: t[1])

        return word_2_distance_list

    def __should_spell_check(self, word):

        is_long_enough = len(word) >= 2
        has_digits = any([c.isdigit() for c in word])
        has_dashes = any([c == "-" for c in word])

        return is_long_enough and not has_digits and not has_dashes

    def correct_spell_check(self, pred):
        pred_words = pred.split(" ")
        corrected_pred_words = []

        for pred_word in pred_words:

            if not self.__should_spell_check(pred_word):
                corrected_pred_words.append(pred_word)
                continue

            words_with_matching_ngrams = self.__get_words_with_matching_ngrams(pred_word)
            results_list = self.__get_close_matches(pred_word, words_with_matching_ngrams,
                                                    max_distance=self.MAX_DISTANCE)

            if len(results_list) == 0:
                corrected_pred_words.append(pred_word)
                continue

            added_word = False
            for distance in range(self.MAX_DISTANCE + 1):

                results_list_by_distance = [(word, dist) for word, dist in results_list if dist == distance]

                if len(results_list_by_distance) == 0:
                    continue

                results_2_count = sorted([(r, self.word_2_count[r]) for r, _ in results_list_by_distance],
                                         key=lambda t: -t[1])

                corrected_pred_word, _ = results_2_count[0]

                corrected_pred_words.append(corrected_pred_word)
                added_word = True
                break

            if not added_word:
                corrected_pred_words.append(pred_word)

        return " ".join(corrected_pred_words)
