"""
Visualize metadata. Sign's bounding box and type overlay
"""

import argparse
import os

import apollo_python_common.proto_api as roi_metadata
import visualization
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path",
                        type=str, required=True)
    parser.add_argument("-o", "--output_path",
                        type=str, required=True)
    args = parser.parse_args()
    utils.make_dirs([args.output_path])

    metadata = roi_metadata.read_imageset_file(args.input_path)
    visualization.visualize_metadata(
        os.path.dirname(args.input_path), metadata, args.output_path)


if __name__ == "__main__":
    main()
