from __future__ import print_function
import os
import argparse
from contextlib import contextmanager
import time
import re


@contextmanager
def cwd(path):
    """Change directory to the given path and back
    """
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


def main():
    parser = argparse.ArgumentParser(description='Run Caffe training with given config file.')

    parser.add_argument('-configfile', '--configfile', help='Config file currently running',
                        required=False)
    parser.add_argument('-solver', '--solverfile', help='Prototxt solver file for caffe training',
                        required=False)
    parser.add_argument('-datadir', '--datadir', help='Data directory where the dataset is located',
                        required=True)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=True)
    parser.add_argument('-logdir', '--logdir', help='Log directory', required=True)

    args, extra = parser.parse_known_args()
    args = vars(args)
    print("Arguments: %s" % args)
    if extra:
        print('Extra arguments: "%s" (will be passed to caffe as-is)' % " ".join(extra))

    model_path = args['outputdir']
    if not model_path or not os.path.isdir(model_path):
        raise RuntimeError("Output directory %s does not exist" % model_path)
    log_dir = args['logdir']
    if not log_dir or not os.path.isdir(log_dir):
        raise RuntimeError("Log directory %s does not exist" % log_dir)
    data_path = args['datadir']
    if not data_path or not os.path.isdir(data_path):
        raise RuntimeError("Data directory %s does not exist" % data_path)

    max_iter_pattern = re.compile(r'^\s*max_iter:\s(?P<MAX_ITER>\d+)\s*$')
    iter_loss_pattern = re.compile(
        r'^.*\sIteration\s+(?P<ITERATION>\d+)\s.*loss\s=\s(?P<LOSS>[-+]?\d*.?\d+(e[-+]\d+)?)(\s.*)?$')

    with cwd(data_path):
        s = time.time()
        f = None
        max_iter = None
        progress = 1
        loss = 1
        stdout_idx = 0
        # Pipe caffe output like `| ~/stdout.txt` for Philly to see the progress
        base_path = os.environ.get('PHILLY_HOME', os.environ.get('HOME', '/tmp'))
        stdout_file = os.path.join(base_path, 'stdout.txt')
        while os.path.exists(stdout_file):
            stdout_idx += 1
            stdout_file = os.path.join(base_path, 'stdout_%d.txt' % stdout_idx)
        exit_file = os.path.join(base_path, 'exit')
        # `touch ~/exit` to exit the job
        while not os.path.exists(exit_file):
            if os.path.exists(stdout_file):
                if not f:
                    # noinspection PyBroadException
                    try:
                        f = open(stdout_file)
                    except:
                        print("stdout removed")
            else:
                if f:
                    f.close()
                f = None
            if f:
                # if there is any stdout, Tee it
                line = f.readline()
                if not line:
                    time.sleep(1)
                    if time.time() - s > 60:
                        s = time.time()
                        print("PROGRESS: {}%".format(progress))
                        print("EVALERR: {}%".format(loss))
                    continue
                line = line.strip()
                print(line)
                if not line:
                    continue
                if not max_iter:
                    # Another chance to find max_iter
                    m = re.match(max_iter_pattern, line)
                    if m:
                        max_iter = int(m.group('MAX_ITER'))
                if not max_iter:
                    continue
                m = re.match(iter_loss_pattern, line)
                if m:
                    progress = float(m.group('ITERATION')) / max_iter * 100
                    loss = float(m.group('LOSS'))
                    print("PROGRESS: {}%".format(progress))
                    print("EVALERR: {}%".format(loss))
                continue

            time.sleep(10)
            if time.time() - s > 10:
                s = time.time()
                print("PROGRESS: {}%".format(progress))
                print("EVALERR: {}%".format(loss))
        if f:
            f.close()
    print("PROGRESS: {}%".format(100))


if __name__ == '__main__':
    main()
