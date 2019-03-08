from ocr.scripts.spell_checker import SpellChecker


class TextCorrector:

    def __init__(self, spell_checker_resources_path):
        self.spell_checker = None if spell_checker_resources_path is None else SpellChecker(
            spell_checker_resources_path)

    def __remove_double_dash(self, text):
        return ''.join([j for i, j in enumerate(text) if j != "-" or (j == "-" and j not in text[i - 1:i])])

    def __remove_double_space(self, text):
        return ''.join([j for i, j in enumerate(text) if j != " " or (j == " " and j not in text[i - 1:i])])

    def __standardize_dashes(self, text):
        return text.replace("—", "-")

    def __merge_highway_identifiers(self, text):
        words = text.split(" ")
        digit_words_indexes = [i for i, word in enumerate(words) if word.isdigit()]

        if len(digit_words_indexes) != 1:
            return text

        digit_index = digit_words_indexes[0]

        if digit_index != len(words) - 2:
            return text

        if len(words[digit_index + 1]) != 1:
            return text

        pre_words = " ".join(words[:digit_index])
        merged_digits = words[digit_index] + words[digit_index + 1]

        new_value = pre_words + " " + merged_digits
        return new_value.strip()

    # 11/2 -> 1 1/2
    def __fix_distance_text(self, s):
        words = [w for w in s.split(" ") if "/" in w]
        if len(words) == 0:
            return s

        word = words[0]
        sides = word.split("/")
        left_side, right_side = sides[0], sides[1]
        if len(left_side) == 2:
            new_word = "{} {}/{}".format(left_side[0], left_side[1], right_side)
        else:
            new_word = word

        return s.replace(word, new_word)

    def __is_number(self, s):
        return all([c.isdigit() for c in s])

    # 14 mile => 1/4 mile
    def __add_missing_slash(self, s):
        words = s.split(" ")

        if len(words) != 2:
            return s

        first_word, second_word = words[0], words[1]

        if second_word != "mile":
            return s

        if not self.__is_number(first_word) or len(first_word) != 2:
            return s

        new_word = "{}/{} {}".format(first_word[0], first_word[1], second_word)
        return new_word

    # 1/z -> 1/2
    # a/4 -> 1/4
    def __correct_slash_words(self, s):
        words = [w for w in s.split(" ") if "/" in w]
        if len(words) != 1:
            return s

        word = words[0]
        halves = word.split("/")

        if len(halves) != 2:
            return s

        first_half, second_half = halves[0], halves[1]
        first_half = first_half if first_half.isdigit() else 1
        second_half = second_half if second_half.isdigit() else 2

        new_word = f"{first_half}/{second_half}"

        return s.replace(word, new_word)

    def correct_text(self, pred):
        pred = self.__remove_double_dash(pred)
        pred = self.__remove_double_space(pred)
        pred = self.__standardize_dashes(pred)
        pred = self.__merge_highway_identifiers(pred)
        pred = self.__fix_distance_text(pred)
        pred = self.__add_missing_slash(pred)
        pred = self.__correct_slash_words(pred)

        if self.spell_checker is not None:
            pred = self.spell_checker.correct_spell_check(pred)

        return pred
