from os import listdir
from os.path import isfile, join, splitext
from random import shuffle
import shutil
import os
import apollo_python_common.io_utils as io_utils
import argparse
from apollo_python_common.io_utils import IMG_EXTENSIONS


def get_train_test_images(all_imgs_path, train_test_threshold):
    img_files = [f for f in listdir(all_imgs_path) if isfile(join(all_imgs_path, f)) and splitext(f)[1] in IMG_EXTENSIONS]
    all_trips_files = [(f.split('_')[0], f) for f in img_files]
    trips_files = dict()
    for trip, file in all_trips_files:
        files = trips_files.get(trip, [])
        files.append(file)
        trips_files[trip] = files
    trips_list = list(trips_files.keys())
    print([(k, len(v)) for k, v in trips_files.items()])
    shuffle(trips_list)
    train_trips = set()
    count_train = 0
    for t in trips_list:
        count_train += len(trips_files[t])
        train_trips.add(t)
        if float(count_train) / len(img_files) >= train_test_threshold:
            break
    test_trips = set(trips_list).difference(train_trips)
    train_files =[trip_file for trip in train_trips for trip_file in trips_files[trip]]
    test_files =[trip_file for trip in test_trips for trip_file in trips_files[trip]]
    return train_files, test_files


def copy_files_to_folder(all_imgs_path, source_files, dest_folder):
    if os.path.isdir(dest_folder):
        shutil.rmtree(dest_folder)
    io_utils.create_folder(dest_folder)
    for file in source_files:
        shutil.copyfile(os.path.join(all_imgs_path, file), os.path.join(dest_folder, file))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_folder', type=str, required=True,
                        help='The folder where all images are located')
    parser.add_argument('--train_folder', type=str, required=True,
                        help='The folder where train images should be stored')
    parser.add_argument('--test_folder', type=str, required=True,
                        help='The folder where test images should be stored')
    parser.add_argument('--train_test_threshold', type=float, required=True,
                        help='Train test split threshold (e.g. 0.7 for 70%)')
    args = vars(parser.parse_args())
    images_folder = args['images_folder']
    train_folder = args['train_folder']
    test_folder = args['test_folder']
    train_test_threshold = args['train_test_threshold']
    train_files, test_files = get_train_test_images(images_folder, train_test_threshold)
    print(('Count train files:', len(train_files), round(100*float(len(train_files)) / (len(train_files) + len(test_files)), 2), '%'))
    print(('Count test files:', len(test_files), round(100*float(len(test_files)) / (len(train_files) + len(test_files)), 2), '%'))
    copy_files_to_folder(images_folder, train_files, train_folder)
    copy_files_to_folder(images_folder, test_files, test_folder)



