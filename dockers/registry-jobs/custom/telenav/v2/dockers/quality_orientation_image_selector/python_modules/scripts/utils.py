import os
import cv2
import numpy as np
from random import randint, shuffle

def blend_transparent(source_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (source_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


def grayscale(rgb):
    return rgb.dot([0.299, 0.587, 0.114])


def saturation(rgb):
    gs = grayscale(rgb)
    alpha = 2 * np.random.random() * 0.5
    alpha += 1 - 0.5
    rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
    return np.clip(rgb, 0, 255)


def brightness(rgb):
    alpha = 2 * np.random.random() * 0.5
    alpha += 1 - 0.7
    rgb = rgb * alpha
    return np.clip(rgb, 0, 255)


def contrast(rgb):
    gs = grayscale(rgb).mean() * np.ones_like(rgb)
    alpha = 2 * np.random.random() * 0.5
    alpha += 1 - 0.5
    rgb = rgb * alpha + (1 - alpha) * gs
    return np.clip(rgb, 0, 255)


def lighting(img):
    cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov)
    noise = np.random.randn(3) * 0.5
    noise = eigvec.dot(eigval * noise) * 255
    img += noise
    return np.clip(img, 0, 255)


def random_transform(img):
    img = img.astype('float32')
    color_jitter = []
    # color_jitter.append(saturation)
    color_jitter.append(brightness)
    color_jitter.append(contrast)
    color_jitter.append(lighting)
    shuffle(color_jitter)
    for jitter in color_jitter:
        img = jitter(img)
    return img
