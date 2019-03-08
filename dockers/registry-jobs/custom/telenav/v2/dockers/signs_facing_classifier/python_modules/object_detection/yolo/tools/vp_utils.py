import math


def get_image_fc_vp(vp_y, vp_confidence, detector, image):
    if vp_confidence > detector.VP_CONFIDENCE_THRESHOLD:
        crop_y = math.floor(vp_y * detector.VP_SIGNIFICATIVE_Y_PERCENTAGE)
        new_image = image[:crop_y, :]
        return new_image
    else:
        return image
