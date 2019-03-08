"""
Converts .blob image into .npy image
"""

import argparse
import network_setup

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_blob",
                        type=str, required=True)
    parser.add_argument("-o", "--output_npy",
                        type=str, required=True)
    args = parser.parse_args()
    network_setup.convert_mean(args.input_blob, args.output_npy)


if __name__ == "__main__":
    main()
