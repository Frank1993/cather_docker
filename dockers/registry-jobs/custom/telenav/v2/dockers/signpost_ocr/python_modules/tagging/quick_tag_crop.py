import os
import cv2

import apollo_python_common.io_utils as io_utils
import pandas as pd


def show_image(img, label, coords):
    cv2.namedWindow("image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    draw_roi(img, label, coords)
    cv2.imshow("image", img)
    pressed_key = cv2.waitKey()
    return pressed_key


def is_valid_command(pressed_key):
    return pressed_key in key_ord_2_class.keys()


def get_target_folder_for_key_press(pressed_key):
    return os.path.join(output_dir, key_2_class[key_ord_2_key[pressed_key]])


def get_roi_crop_name(img_name, coords, label):
    crop_name = img_name.split('.')[0]

    for coord in coords:
        crop_name = crop_name + '_' + str(coord)

    return '{}_{}.jpg'.format(crop_name, label)


def log_action(pressed_key, valid_command_received):
    if not valid_command_received:
        print("Invalid command. Must press one of the following keys {}".format(key_2_class.keys()))
        return

    target_folder = get_target_folder_for_key_press(pressed_key)
    print("Pressed: {} \nMoving img to {} ({})".format(key_ord_2_key[pressed_key], target_folder,
                                                         len(os.listdir(target_folder))))


def draw_roi(img, label, coords):
    color = (0, 255, 0)
    cv2.rectangle(img, (coords[0], coords[1]), (coords[2] + 1, coords[3] + 1), color, 2)
    cv2.putText(img, label, (coords[0], coords[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def save_crop(img, coords, crop_path):
    crop_img = img[coords[1]: coords[3], coords[0]: coords[2]]

    print('\nWriting cropped image: {}', crop_path)
    cv2.imwrite(crop_path, crop_img)


def quit_and_save_progress(pressed_key, progress_df):
    if pressed_key == 'q':
        print('quit key pressed.')
        progress_df.to_csv(progress_file)
        return True
    else:
        return False


def get_df_row(progress_header, row):
    csv_row = []
    for item in progress_header:
        csv_row.append(row[item])

    print(csv_row)
    return csv_row


def label_images(sample_df, progress_index):
    for idx, row in sample_df[progress_index:].iterrows():
        print('idx: ', idx)
        image_path = os.path.join(input_dir, row['image'])
        print("\nRead: {}".format(os.path.basename(image_path)))
        img = cv2.imread(image_path)
        if img is None:
            print('\nRead image with errors: {}'.format(row['image']))
            continue

        valid_command_received = False
        coords = row['tl_col'], row['tl_row'], row['br_col'], row['br_row']
        print('bbox coordinates: ', coords)

        while not valid_command_received:
            pressed_key = show_image(img.copy(), row['roi_class'], coords)
            valid_command_received = is_valid_command(pressed_key)
            log_action(pressed_key, valid_command_received)

            if valid_command_received:
                dst_folder = get_target_folder_for_key_press(pressed_key)
                save_crop(img, coords,
                          os.path.join(dst_folder, get_roi_crop_name(row['image'], coords, row['roi_class'])))
                update_progress_index(idx + 1)


def create_dest_folders():
    for _, label_class in key_2_class.items():
        dir_name = os.path.join(output_dir, label_class)
        os.makedirs(dir_name, exist_ok=True)
        print("Created folder: {}".format(dir_name))


def update_progress_index(progress_index):
    prog_fh = open(progress_file, 'w')
    prog_fh.write(str(progress_index))
    prog_fh.flush()
    prog_fh.close()


def load_progress_index():
    if os.path.isfile(progress_file):
        prog_fh = open(progress_file, 'r')
    else:
        prog_fh = open(progress_file, 'a+')

    progress_index = prog_fh.read()
    print('load_progress_index {} :'.format(progress_index))

    prog_fh.close()
    if progress_index == '':
        progress_index = '0'

    return int(progress_index)


def run():
    create_dest_folders()

    sample_df = pd.read_csv(sample_file)
    progress_index = load_progress_index()

    print('progress index {}:'.format(progress_index))

    label_images(sample_df, progress_index)


def read_config_json(json_path):
    """ Reads the json config file and returns the config options.
        The config options are:
            - images_input_dir is the images directory inside the base directory
            - tag_metadata_file is the csv file containing the list of images to be tagged
            - mapping is the keyboard mapping of keys for the quick tagging process
    """
    data = io_utils.json_load(json_path)

    in_dir = data['input_dir']
    out_dir = data['output_dir']
    smp_file = data['sample_file']
    prog_file = data['progress_file']
    mapping = data['mapping']

    return in_dir, out_dir, smp_file, prog_file, mapping


if __name__ == '__main__':
    input_dir, output_dir, sample_file, progress_file, key_2_class = read_config_json("./crop_tagging_config.json")

    key_ord_2_class = {ord(key): value for key, value in key_2_class.items()}
    key_ord_2_key = {ord(key): key for key, _ in key_2_class.items()}

    run()
