"""
 Clean up rois outside image.
 Clean up rois small than threshold.
 Clean up missing images.
 Clean up images with bad aspect ratio.
"""

import argparse
import os
import logging

import apollo_python_common.log_util as log_util

import dataset
import apollo_python_common.proto_api as proto_api



def main():

    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    logger.info('Clean dataset')

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path",
                        type=str, required=True)
    parser.add_argument("-o", "--output_path",
                        type=str, required=True)
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    metadata = dataset.load_valid_rois(args.input_path)
    proto_api.serialize_proto_instance(metadata, args.output_path)



if __name__ == "__main__":
    main()
