import os
import time
import pandas as pd

import apollo_python_common.proto_api as meta

from absl import flags
from absl import app
from scripts.rois_data import RoisLabels

flags.DEFINE_string('rois_file', '', 'The ROIs file to be used for batching.')
flags.DEFINE_string('output_dir', '', 'The output directory where the csv batch files will be saved.')
flags.DEFINE_boolean('csv_file', False, 'The ROIs input file is in CSV format - in case of doing after batches')
flags.DEFINE_integer('num_batches', 1, 'Number of ROI batches to make.')
flags.DEFINE_integer('num_rois', 5000, 'Number of ROIs in each batch.')
flags.DEFINE_integer('num_classes', 55, 'Number of ROI classes - this will allow for balanced class selection.')


FLAGS = flags.FLAGS


def make_roi_dataframe(rois_file):
    """ Given a roi protobuf file, it creates a randomized dataframe with all the rois in the file. """

    rois_labels = RoisLabels(rois_file)

    data_dict = {'image': [], 'tl_row': [], 'tl_col': [], 'br_row': [], 'br_col': [], 'roi_class': []}

    for img in rois_labels.rois_dict.keys():
        rois = rois_labels.rois_dict[img]
        for roi in rois:
            data_dict['image'].append(img)
            data_dict['tl_row'].append(roi.rect.tl.row)
            data_dict['tl_col'].append(roi.rect.tl.col)
            data_dict['br_row'].append(roi.rect.br.row)
            data_dict['br_col'].append(roi.rect.br.col)
            data_dict['roi_class'].append(meta.get_roi_type_name(roi.type))

    roi_df = pd.DataFrame(data_dict)
    roi_df = roi_df.sample(frac=1).reset_index(drop=True)

    return roi_df


def load_roi_dataframe(csv_file):
    """ Loads a CSV file of ROI's as a dataframe. """
    return pd.read_csv(filepath_or_buffer=csv_file)


def drop_items_from_df(input_df, items_to_drop):
    """ Given an input dataframe removes the items_to_drop from it. """
    combined_df = input_df.append(items_to_drop)
    return combined_df[~combined_df.index.duplicated(keep=False)]


def make_roi_sample(roi_df, num_rois, num_classes):
    """ Given the roi dataframe, the number of rois and the number of available classes, returns a sample of num_rois
        from the original dataframe, with a balanced number of classes. This means the data frame is grouped by
        classes and we return a number of max num_rois/num_classes of each class if available. The rest up to num_rois
        is filled with a random subsample of the original dataframe.
    """
    rois_per_class = num_rois // num_classes

    print('rois per class: ', rois_per_class)

    grouped_df_list = []
    for roi_class, grouped_df in roi_df.groupby('roi_class'):
        subset_df = grouped_df[:rois_per_class]
        grouped_df_list.append(subset_df)

    roi_sample_df = pd.concat(grouped_df_list)
    sample_size = len(roi_sample_df)

    roi_df = drop_items_from_df(roi_df, roi_sample_df)
    remainder_sample = roi_df.sample(n=num_rois - sample_size)

    roi_sample_df = pd.concat([roi_sample_df, remainder_sample])
    roi_df = drop_items_from_df(roi_df, remainder_sample)

    print('roi sample class value count: ', roi_sample_df['roi_class'].value_counts())
    print('roi sample length: ', len(roi_sample_df))
    print(roi_sample_df.head())

    return roi_df, roi_sample_df


def make_roi_sample_batches(roi_df, num_batches, num_rois, num_classes, output_dir):
    """ Given the roi dataframe, it creates num_batches roi samples and saves them each in a CSV file in the given
        output directory.
    """

    in_df_size = len(roi_df)
    print('input roi df size: ', in_df_size)

    csv_file = 'roi_sample{}.csv'
    for i in range(num_batches):
        roi_df, roi_sample_df = make_roi_sample(roi_df, num_rois, num_classes)

        out_df_size = len(roi_df)
        print('out roi_df size, diff: ', out_df_size, in_df_size - out_df_size)
        roi_sample_df = roi_sample_df.sample(frac=1)  # shuffle the sample before writing to csv
        roi_sample_df.to_csv(os.path.join(output_dir, csv_file.format(i)), index=False)

    roi_df.to_csv(os.path.join(output_dir, 'unselected_rois.csv'))


def main(_):
    assert FLAGS.rois_file, '`rois_file` is missing.'
    assert FLAGS.output_dir, '`output_dir` is missing.'
    assert FLAGS.num_classes, '`num_classes` is missing.'

    if FLAGS.csv_file:
        roi_df = load_roi_dataframe(FLAGS.rois_file)
    else:
        roi_df = make_roi_dataframe(FLAGS.rois_file)

    make_roi_sample_batches(roi_df, FLAGS.num_batches, FLAGS.num_rois, FLAGS.num_classes, FLAGS.output_dir)


if __name__ == '__main__':
    app.run(main)
