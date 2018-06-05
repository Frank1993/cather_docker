#!/usr/bin/python
import sys
import os
import shutil
import argparse

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataDir', help='Currently not used')
    parser.add_argument('--modelDir', help='Output dir')
    parser.add_argument('--logDir', help='Currently not used')
    parser.add_argument('--nGPU', type=int, help='number of GPU for this process')
    parser.add_argument('--prevModelPath', help='The prev model path should be the same as modelDir, otherwise, copy everything into modelDir')
    parser.add_argument('--yamlConfigFile', help='The YAML file containing detectron training setup, should be in the same folder as this script')
    return parser.parse_args(argv)

def main(argv):
    args = parse_args(argv)

    # Must import train_net after os.environ is modified
    from train_net import main as train_net

    # Copy previous result to output directory so it can be loaded if it exists
    if args.prevModelPath and args.prevModelPath != 'NONE':
        for item in os.listdir(args.prevModelPath):
            s = os.path.join(args.prevModelPath, item)
            d = os.path.join(args.modelDir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)

    base_dir = os.path.dirname(os.path.realpath(__file__))
    train_net(['--cfg', args.yamlConfigFile, 'OUTPUT_DIR', args.modelDir])

if __name__ == '__main__':
    main(sys.argv[1:])
