from math import sqrt, ceil
import numpy as np


class Rectangle:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    @staticmethod
    def initialize_with_proto_rect(proto_rect):
        return Rectangle(proto_rect.tl.col, proto_rect.tl.row, proto_rect.br.col, proto_rect.br.row)

    def get_intersection_area(self, sec_rect):
        rect_a = self
        dx = min(rect_a.xmax, sec_rect.xmax) - max(rect_a.xmin, sec_rect.xmin)
        dy = min(rect_a.ymax, sec_rect.ymax) - max(rect_a.ymin, sec_rect.ymin)
        if (dx >= 0) and (dy >= 0):
            return float(dx * dy)
        else:
            return float(0)

    def area(self):
        return self.width() * self.height()

    def width(self):
        return float(self.xmax - self.xmin)

    def height(self):
        return float(self.ymax - self.ymin)

    def contains_rectangle(self, rect):
        return self.xmax >= rect.xmax and self.xmin <= rect.xmin and self.ymin <= rect.ymin and self.ymax >= rect.ymax

    def get_overlapped_rect(self, sec_rect):
        rect_a = self
        x_min = max(rect_a.xmin, sec_rect.xmin)
        y_min = max(rect_a.ymin, sec_rect.ymin)
        x_max = min(rect_a.xmax, sec_rect.xmax)
        y_max = min(rect_a.ymax, sec_rect.ymax)
        if x_min <= x_max and y_min <= y_max:
            return Rectangle(x_min, y_min, x_max, y_max)
        else:
            return Rectangle(0, 0, 0, 0)

    def get_distance_between_centers(self, sec_rect):
        center_1_x = (self.xmin + self.xmax) / 2
        center_1_y = (self.ymin + self.ymax) / 2
        center_2_x = (sec_rect.xmin + sec_rect.xmax) / 2
        center_2_y = (sec_rect.ymin + sec_rect.ymax) / 2
        dist = sqrt((center_2_x - center_1_x)**2 + (center_2_y - center_1_y)**2)
        return dist

    def get_bounding_box_rect(self, sec_rect):
        first_rect = self
        return Rectangle(min(first_rect.xmin, sec_rect.xmin),
                         min(first_rect.ymin, sec_rect.ymin),
                         max(first_rect.xmax, sec_rect.xmax),
                         max(first_rect.ymax, sec_rect.ymax))

    def get_average_box_rect(self, sec_rect):
        first_rect = self
        return Rectangle(np.mean([first_rect.xmin, sec_rect.xmin]),
                         np.mean([first_rect.ymin, sec_rect.ymin]),
                         np.mean([first_rect.xmax, sec_rect.xmax]),
                         np.mean([first_rect.ymax, sec_rect.ymax]))

    def intersection_over_union(self, sec_rec):
        intersect_area = self.get_intersection_area(sec_rec)
        union_area = self.area() + sec_rec.area() - intersect_area
        if union_area != 0:
            ret_val = intersect_area / union_area
        else:
            ret_val = 0
        return float(ret_val)

    def clip_in_rect(self, in_rect):
        """ Given a larger rectangle, clip this rectangle inside the bounds of the outer larger rectangle represented by
         in_rect. """
        if self.get_intersection_area(in_rect) <= 0:
            raise ValueError("current rectangle needs to intersect with the input rectangle.")

        clp_xmin = max(in_rect.xmin, self.xmin)
        clp_xmax = min(in_rect.xmax, self.xmax)
        clp_ymin = max(in_rect.ymin, self.ymin)
        clp_ymax = min(in_rect.ymax, self.ymax)

        return Rectangle(clp_xmin, clp_ymin, clp_xmax, clp_ymax)

    def expand_to_square(self, in_rect):
        """
        Expands this rectangle to a square. When the square has bounds outside of the in_rect rectangle, the padding
        is added inside the bounds.

        :param in_rect: the input rectangle which is used to check the bounds of this square
        :return: the corresponding square
        """
        if self.get_intersection_area(in_rect) <= 0:
            raise ValueError("current rectangle needs to intersect with the input rectangle.")

        pad_size = ceil(abs((self.height() - self.width()) / 2))
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax

        if self.width() < self.height():

            xmin = max(0, self.xmin - pad_size)
            xmax = min(in_rect.xmax, self.xmax + pad_size)

            if xmin == 0:
                xmax += pad_size * 2
                xmin = self.xmin

            if xmax == in_rect.xmax:
                xmin -= pad_size * 2
                xmax = self.xmax

        elif self.width() > self.height():
            ymin = max(0, self.ymin - pad_size)
            ymax = min(in_rect.ymax, self.ymax + pad_size)

            if ymin == 0:
                ymax += pad_size * 2
                ymin = self.ymin

            if ymax == in_rect.ymax:
                ymin -= pad_size * 2
                ymax = self.ymax

        return Rectangle(xmin, ymin, xmax, ymax)

    def __mul__(self, factor):
        center_x = (self.xmin + self.xmax) // 2
        center_y = (self.ymin + self.ymax) // 2

        new_xmin = center_x - int(factor * self.width() / 2)
        new_xmax = center_x + int(factor * self.width() / 2)
        new_ymin = center_y - int(factor * self.height() / 2)
        new_ymax = center_y + int(factor * self.height() / 2)

        new_rect = Rectangle(new_xmin, new_ymin, new_xmax, new_ymax)

        return new_rect

    def __repr__(self):
        return 'xmin {}, ymin {}, xmax {}, ymax {}'.format(self.xmin, self.ymin, self.xmax, self.ymax)

