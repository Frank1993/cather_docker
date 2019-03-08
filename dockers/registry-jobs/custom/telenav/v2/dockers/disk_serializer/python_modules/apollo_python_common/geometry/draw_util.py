import colorsys
import cv2
import os
import apollo_python_common.proto_api as proto_api


def get_n_different_colors(nr_of_colors):
    rgb_tuples = [colorsys.hsv_to_rgb(x * 1.0 / nr_of_colors, 0.5, 0.5) for x in range(nr_of_colors)]
    return [(r*255, g*255, b*255) for (r, g, b) in rgb_tuples]


def draw_rois_to_file(raw_imgs_folder, raw_image_file_name, file_rois, out_folder):
    img_path = os.path.join(raw_imgs_folder, raw_image_file_name)
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    for roi in file_rois:
        label_name, rect = proto_api.get_rect_from_roi(roi)
        color = (0, 255, 0)
        cv2.rectangle(image, (int(rect.xmin), int(rect.ymin)), (int(rect.xmax) + 1, int(rect.ymax) + 1), color, 2)
        img_text = label_name
        cv2.putText(image, img_text, (int(rect.xmin), int(rect.ymin)-3), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    out_file_name = os.path.join(out_folder, raw_image_file_name)
    cv2.imwrite(out_file_name, image)
