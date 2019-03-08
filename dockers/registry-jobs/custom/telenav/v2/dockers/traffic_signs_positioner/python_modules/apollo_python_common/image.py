import cv2
import numpy as np
import scipy.misc
from PIL import Image
import os
import requests
from collections import namedtuple
import urllib.parse
import requests


REQUEST_TIMEOUT_IN_SECONDS = 3.0
HTTP_SCHEMAS = ['http', 'https']
OscDetails = namedtuple('OscDetails', ['image_id', 'osc_api_url'], verbose=False)


def __get_image_meta(image_path):
    # PIL is fast for metadata
    return Image.open(image_path)


def get_size(image_path):
    image = __get_image_meta(image_path)
    return image.width, image.height


def get_aspect_ratio(image_path):
    image = __get_image_meta(image_path)
    return image.width / image.height


def get_bgr(image_url, osc_details=None):
    parsed_url = urllib.parse.urlparse(image_url)
    if _url_resource_exists(image_url, parsed_url.scheme):
        image = _read_image(image_url, parsed_url.scheme)
        if image is None:
            raise Exception("The content of image {} is invalid.".format(image_url))
        else:
            return image
    else:
        if osc_details is None:
            raise Exception("Image {} is missing.".format(image_url))
        else:
            new_image_url = __get_image_path(osc_details.image_id, osc_details.osc_api_url)
            if new_image_url is None:
                raise Exception("Image {} with id {} is missing.".format(image_url, osc_details.image_id))
            else:
                return get_bgr(new_image_url)


def _http_resource_exists(path):
    r = requests.head(path)
    return r.status_code == requests.codes.ok


def _file_exists(path):
    return os.path.isfile(path)


def _url_resource_exists(url, scheme):
    if scheme in HTTP_SCHEMAS:
        return _http_resource_exists(url)
    elif scheme == "":
        return _file_exists(url)
    else:
        raise Exception("The schema {} for image url {} is not allowed.".format(scheme, url))


def _read_image(url, scheme):
    if scheme in HTTP_SCHEMAS:
        return _read_image_http(url)
    elif scheme == "":
        return _read_image_file(url)
    else:
        raise Exception("The schema {} for image url {} is not allowed.".format(scheme, url))


def _read_image_file(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR


def _read_image_http(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        response.raw.decode_content = True
        image = np.fromstring(response.content, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)  # BGR order
        return image
    else:
        return None


def get_rgb(image_path, osc_details=None):
    image_bgr = get_bgr(image_path, osc_details)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def rotate_image(image, angle):
    height, width, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
    result = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
    return result


def flip_image(img):
    return cv2.flip(img, 1)


def resize_image_fill(image, height, width, channels, interpolation='bilinear'):
    """
    Resize image by filling with black
    """
    horizontal_padding = 0
    vertical_padding = 0
    if image.shape[0] == height and image.shape[1] == width:
        return image, 0, 0
    width_ratio = float(image.shape[1]) / width
    height_ratio = float(image.shape[0]) / height
    if width_ratio > height_ratio:
        resize_width = width
        resize_height = int(round(image.shape[0] / width_ratio))
        if (height - resize_height) % 2 == 1:
            resize_height += 1
    else:
        resize_height = height
        resize_width = int(round(image.shape[1] / height_ratio))
        if (width - resize_width) % 2 == 1:
            resize_width += 1
    image = scipy.misc.imresize(image, (resize_height, resize_width), interpolation)

    if width_ratio > height_ratio:
        padding = (height - resize_height) // 2
        noise_size = (padding, width)
        if channels > 1:
            noise_size += (channels,)
        noise = np.zeros(noise_size).astype('uint8')
        image = np.concatenate((noise, image, noise), axis=0)
        vertical_padding = padding
    else:
        padding = (width - resize_width) // 2
        noise_size = (height, padding)
        if channels > 1:
            noise_size += (channels,)
        noise = np.zeros(noise_size).astype('uint8')
        image = np.concatenate((noise, image, noise), axis=1)
        horizontal_padding = padding
    return image, horizontal_padding, vertical_padding


def cv_resize(image, new_width, new_height, default_interpolation=True):
    """ Use AREA for shrinking images with opencv and LINEAR interpolation for zooming. """

    new_size = (new_width, new_height)
    if default_interpolation:
        resized_img = cv2.resize(image, new_size)
    else:
        width = image.shape[1]
        height = image.shape[0]

        if width <= new_width or height <= new_height:
            resized_img = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        else:
            resized_img = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    return resized_img

def __get_image_path(image_id, osc_api_url):
    final_picture_path = None
    r = requests.get('{}{}'.format(osc_api_url, image_id), timeout=REQUEST_TIMEOUT_IN_SECONDS)
    if r.status_code == requests.codes.ok:
        rj = r.json()
        valid_response = "osv" in rj and "photoObject" in rj["osv"] and "path" in rj["osv"]["photoObject"] and \
                         "photoName" in rj["osv"]["photoObject"]
        if valid_response:
            raw_path = rj["osv"]["photoObject"]["path"]
            if '/' in raw_path:
                storage = raw_path.split('/')[0]
                file_location = '/'.join(raw_path.split('/')[1:])
                photo_name = rj["osv"]["photoObject"]["photoName"]
                if len(storage) * len(file_location) * len(photo_name) > 0:
                    final_picture_path = "/mnt/{0}/www/html/open-street-view.skobbler.net/current/{1}/ori/{2}". \
                        format(storage, file_location, photo_name)
    return final_picture_path
