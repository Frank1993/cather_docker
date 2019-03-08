import apollo_python_common.proto_api as meta
from tqdm import tqdm
import os
from PIL import Image
import shutil
from object_detection.roi_ssd.utils.roi_utils import get_rect_from_roi


def get_resolution(image_path):
    return Image.open(image_path).size


if __name__=="__main__":
    images_path = '/data/datasets/st_tr_tl_sl_gw_15_11_2017/'
    roi_file = '/data/datasets/st_tr_tl_sl_gw_15_11_2017/rois.bin'
    output_folder = '/data/datasets/st_tr_tl_sl_gw_15_11_2017_selected'
    print((meta.check_imageset(roi_file)))
    roi_dict = meta.create_images_dictionary(meta.read_imageset_file(roi_file))
    print(('initial files count {}'.format(len(roi_dict))))
    remaining_files = dict(list(roi_dict.items()))
    for file_base_name, rois in tqdm(list(roi_dict.items())):
        source_img_file = os.path.join(images_path, file_base_name)
        if not os.path.isfile(source_img_file):
            print(('Inexistent {}'.format(source_img_file)))
            continue
        img_shape = get_resolution(source_img_file)
        img_width = img_shape[0]
        img_height = img_shape[1]
        for roi in rois:
            _, rect = get_rect_from_roi(roi)
            if rect.width() / img_width < 0.04 or rect.height() / img_height < 0.04:
                remaining_files.pop(file_base_name, None)
                break
    print(('final files count {}'.format(len(remaining_files))))
    # print('copying files...')
    # for file_base_name, rois in tqdm(list(remaining_files.items())):
    #     source_img_file = os.path.join(images_path, file_base_name)
    #     shutil.copy(source_img_file, os.path.join(output_folder, file_base_name))

    new_meta = meta.create_imageset_from_dict(remaining_files)
    meta.serialize_proto_instance(new_meta, output_folder, '_rois')
    print((meta.check_imageset(os.path.join(output_folder, "_rois.bin"))))

