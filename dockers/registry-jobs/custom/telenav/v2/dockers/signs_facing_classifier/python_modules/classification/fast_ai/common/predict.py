import numpy as np
from fastai.imports import torch
from fastai.vision import image as faimg
from fastai.vision import *

from apollo_python_common import image as image_api


def roi_2_id(roi_proto):
    return "{}-{}-{}-{}".format(roi_proto.rect.tl.row,
                                roi_proto.rect.tl.col,
                                roi_proto.rect.br.row,
                                roi_proto.rect.br.col)


def extract_cropped_roi(full_img, roi_proto, out_size, sq_crop_factor):
    """ Given a full image, a roi proto and an output size, it returns a cropped roi as a fast.ai Image, based on the
    roi proto coordinates. """
    roi_id = roi_2_id(roi_proto)

    img = image_api.crop_square_roi(full_img, roi_proto.rect.tl.col, roi_proto.rect.tl.row,
                                    roi_proto.rect.br.col, roi_proto.rect.br.row,
                                    sq_crop_factor=sq_crop_factor)
    roi_img = faimg.Image(convert_img_to_tensor(img)).resize((img.shape[2], out_size, out_size))

    return roi_img, roi_id


def convert_img_to_tensor(rgb_img):
    """ Converts a RGB image given as a numpy array to a PyTorch tensor for use in the fast.ai library. """

    if rgb_img.ndim == 2:
        rgb_img = np.expand_dims(rgb_img, 2)

    rgb_img = np.transpose(rgb_img, (1, 0, 2))
    rgb_img = np.transpose(rgb_img, (2, 1, 0))
    return torch.from_numpy(rgb_img.astype(np.float32, copy=False)).div_(255)


def get_inference_learner(params, labels, backbone_model):
    """ Creates a FastAI learner for inference, using the given parameters. This will have an empty data bunch.
        The params argument needs to be an AttributeDict that contains the transforms and model parameters to be
        loaded into the learner. The labels argument is the list of labels for which we do classification.
        The backbone_model argument is the FastAI backbone used for training the model for which we need to do
        inference.
        Normalization is done using the Imagenet dataset image stats.
        This function returns the FastAI learner created along with the list of labels from the learner in the particular
        order the data bunch creates them internally.
    """
    transforms = get_transforms(max_rotate=params.tfms_max_rotate, max_warp=params.tfms_max_warp,
                                flip_vert=params.tfms_flip_vert, do_flip=params.tfms_do_flip)

    empty_data = ImageDataBunch.single_from_classes(params.model_dir, labels,
                                                    tfms=transforms, padding_mode='zeros',
                                                    bs=params.batch_size,
                                                    size=params.image_size).normalize(imagenet_stats)

    return create_cnn(empty_data, backbone_model).load(params.model_name)
