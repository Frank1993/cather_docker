import numpy as np
import cv2

import orbb_definitions_pb2

MASK_VAL = 30


def visualize_mask(img, mask, configuration):
    img_out = img.transpose(1, 2, 0).astype(np.uint8).copy()
    # darken segmented area
    row = 10
    col = 10
    for _, v in configuration.type2color.items():
        mask_blend = mask[..., np.newaxis].transpose(0, 1, 2).astype(np.uint8)
        mask_blend[mask_blend == v] = MASK_VAL

        # draw contours of segmented areas
        _, thresh = cv2.threshold(mask_blend, MASK_VAL - 1, 255, 0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_out, contours, -1,
                         configuration.palette[v*3:v*4], 1)
        cv2.putText(img_out, configuration.label_names[v], (col, row),
                    cv2.FONT_HERSHEY_PLAIN, 1,
                    configuration.palette[v*3:v*4], 2)
        row = row + 10
    return img_out


def visualize_rects(img, class_rects, configuration):
    img_out = img.transpose(1, 2, 0).astype(np.uint8).copy()
    row = 10
    col = 10
    for _, v in configuration.type2color.items():
        cv2.putText(img_out, configuration.label_names[v], (col, row),
                    cv2.FONT_HERSHEY_PLAIN, 1,
                    configuration.palette[v*3:v*4], 2)
        row = row + 10

    for k, v in class_rects.items():
        for r in v:
            cv2.rectangle(img_out, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]),
                          configuration.palette[(k+1)*3:(k+1)*4], 2)

    return img_out


def visualize_metadata(input_path, metadata, output_path):
    # TODO: use configuration
    for ir in metadata.images:
        img_name = os.path.basename(ir.metadata.image_path)
        print(img_name)
        image = cv2.imread(input_path + "/" + img_name)
        image_out = image.copy()
        for roi in ir.rois:
            cv2.rectangle(image_out, (roi.rect.tl.col, roi.rect.tl.row),
                          (roi.rect.br.col, roi.rect.br.row), [255, 0, 0], 2)
            cv2.putText(image_out, orbb_definitions_pb2.Mark.Name(roi.type),
                        (roi.rect.br.col, roi.rect.br.row),
                        cv2.FONT_HERSHEY_PLAIN, 1, [255, 0, 0], 2)

        cv2.imwrite(output_path + "/" + img_name, image_out)