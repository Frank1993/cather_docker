import glob
import os
import cv2
import shutil
import json

RESIZE_WIDTH = 1024
RESIZE_HEIGHT = 720


def click_and_crop(event, x, y, flags, img):
    global ref_pt_list

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_pt_list = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        ref_pt_list.append((x, y))

        resized_img = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT))
        cv2.rectangle(resized_img, ref_pt_list[0], ref_pt_list[1], (0, 255, 0), 2)
        cv2.imshow("image", resized_img)

        height, width, _ = img.shape
        height_offset = height / RESIZE_HEIGHT
        width_offset = width / RESIZE_WIDTH

        roi = img[int(ref_pt_list[0][1] * height_offset):int(ref_pt_list[1][1] * height_offset),
              int(ref_pt_list[0][0] * width_offset):int(ref_pt_list[1][0] * width_offset)]

        cv2.imshow("roi", cv2.resize(roi, (500, 500)))


def show_image(img):
    resized_img = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT))

    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("image", click_and_crop, param=img)

    cv2.imshow("image", resized_img)

    pressed_key = cv2.waitKey()
    return pressed_key


def is_valid_command(pressed_key):
    return pressed_key in key_ord_2_class.keys()


def get_target_folder_for_key_press(pressed_key):
    return os.path.dirname(input_folder) + "/" + key_2_class[key_ord_2_key[pressed_key]]


def log_action(pressed_key, valid_command_received):
    if not valid_command_received:
        print("Invalid command. Must press one of the following keys {}".format(key_2_class.keys()))
        return

    target_folder = get_target_folder_for_key_press(pressed_key)
    print("Pressed: {} \nMoving img to {} ({})".format(key_ord_2_key[pressed_key], target_folder,
                                                       len(os.listdir(target_folder))))


def label_images(images_path):
    for image_path in images_path:
        print("\nRead: {}".format(os.path.basename(image_path)))
        img = cv2.imread(image_path)

        valid_command_received = False
        while not valid_command_received:
            pressed_key = show_image(img)
            valid_command_received = is_valid_command(pressed_key)
            log_action(pressed_key, valid_command_received)

            if valid_command_received:
                dst_folder = get_target_folder_for_key_press(pressed_key)
                shutil.move(image_path, dst_folder)


def create_dest_folders():
    base_folder = os.path.dirname(input_folder)

    for _, label_class in key_2_class.items():
        dir_name = base_folder + "/" + label_class
        os.makedirs(dir_name, exist_ok=True)
        print("Created folder: {}".format(dir_name))


def run():
    images_path = sorted(glob.glob(input_folder + "/*"))

    create_dest_folders()

    label_images(images_path)


def read_json(json_path):
    with open(json_path) as json_data_file:
        data = json.load(json_data_file)

    inp_folder = data['input_folder']
    mapping = data['mapping']

    return inp_folder, mapping


if __name__ == '__main__':
    input_folder, key_2_class = read_json("./tagging_config.json")

    key_ord_2_class = {ord(key): value for key, value in key_2_class.items()}
    key_ord_2_key = {ord(key): key for key, _ in key_2_class.items()}

    run()
