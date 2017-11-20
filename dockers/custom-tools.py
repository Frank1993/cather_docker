#!/usr/bin/env python
from __future__ import print_function

import os
import re
import sys
import json
import shlex
import urllib
import zipfile
import argparse

try:
    from subprocess32 import Popen, PIPE, TimeoutExpired
except ImportError:
    # TODO: Support other ptyhon versions
    print('Error: Currently this script only supports python 2.7')
    print('There are plans in the future to support python 3')
    print('You are using: ' + sys.version)
    exit(1)

# ===========================================================================================
# 
# custom-tools.py
# 
# This script has 3 main functions:
#          
#   build:     Build your custom docker on the local machine.
#   test:      Test the custom docker complies to the requirements.
#   run:       Run a test job on the local machine.
# 
# ===========================================================================================

# Global variables
philly_reg = 'phillyregistry.azurecr.io/philly/'

def find_line(filename, regex):
    """ Finds the first line that matches a regex in the file """
    linenumber = -1
    with open(filename, 'r') as infile:
        for line in infile:
            linenumber += 1;
            if re.search(regex, line):
                return line, linenumber
    return None, linenumber

def command_wrapper(command, pipe=False):
    """ Wrapper function for bash commands """
    command = filter(None, command)
    cmd_str = ' '.join(command)
    print('COMMAND: ' + cmd_str)
    stdOut = 'successful'
    
    if pipe:
        p = Popen(command, stdout=PIPE)
    else:
        p = Popen(command)
        
    try:
        if pipe:
            stdOut, stdErr = p.communicate(timeout=None)
        else:
            p.wait()
    except TimeoutExpired:
        raise TimeoutExpired("%r timed out" % (cmd_str))
    
    if p.returncode != 0:
        raise RuntimeError("%r failed, status code %s" % (cmd_str, p.returncode))        
    
    return stdOut

# Find the executable file
def find_executable(directory):
    executable = None
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)) and filename.startswith('executable'):
            if not executable:
                executable = filename
            else:
                return None
    return executable

# Get FROM line from Dockerfile
def get_from_line(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    for line in lines:
        if line.startswith('FROM'):
            return line
        elif line == '' or line.startswith('#'):
            # Ignore empty or comment lines
            continue
        else:
            # Error out on other lines
            print('\nERROR: First uncommented line must be "FROM*" (' + filename + ')')
            exit(1)

def line_split(line):
    rtn = []
    while len(line) > 0 and ' ' in line:
        index = line.index(' ')
        string = line[0:index]
        line = line[index+1:]
        if '\"' in string and not string.endswith('\"'):
            index = line.find('\"') + 1
            string += ' ' + line[0:index]
            line = line[index+1:]
        if len(string) > 0:
            rtn.append(string)
    if len(line) > 0:
        rtn.append(line)
    return rtn

def add_labels(split_list, label_list):
    for item in split_list:
        split = item.split('=', 1)
        if len(split) == 2:
            split[0] = re.sub(r'^"|"$', '', split[0])
            split[1] = re.sub(r'^"|"$', '', split[1])
            label_list.append(split)
    return label_list

# Get the labels of a dockerfile as a dictionary
def get_labels(filename):
    labels = []
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line: break
            if line.startswith('LABEL '):
                split = line_split(line.rstrip('\n'))
                labels = add_labels(split, labels)
                while split[-1] == '\\':
                    line = f.readline()
                    split = line_split(line.rstrip('\n'))
                    labels = add_labels(split, labels)
    return labels

# Find if a line exists in a file
def line_exists_in_file(search_line, filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    for line in lines:
        if line == search_line:
            return True
    return False

class Tools(object):
    """ Class structure to to create multi layer command inputs """
    def __init__(self):
        print('')
        # Parse the command
        parser = argparse.ArgumentParser(
            description='Custom Docker Tools',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            usage='''custom-tools.py [-h] <command> [<args>]
            
The most commonly used commands are:
   build    - Build your custom docker on the local machine.
   test     - Test the custom docker compiles to the requirements.
   run      - Run a test job on the local machine.''')
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        
        # Check if required tools are installed in the system
        print('Searching for required installed toolkits (outside of python)...')
        toolkits = ['az', 'docker', 'nvidia-docker']
        for toolkit in toolkits:
            if len(command_wrapper(['which', toolkit], True)) == 0:
                print('Missing toolkit: ' + toolkit)
                exit(0)
        
        # Error out if unknown command
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
            
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        
        # Execute the function for the given command
        getattr(self, args.command)()

    def build(self):
        """ Build the target custom docker """
        # Parse the command line
        parser = argparse.ArgumentParser(
            description='Build the target custom docker',
            usage='''custom-tools.py build [-h] [-nc] [-f] [target ...]''',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''
notes:
  > \'target\' is the directory in the \'custom\' directory (registry-jobs/custom/)
  >   Example: bash/v1.0
  > If no \'target\' is given then all the base and toolkit dockers will be build.
  > You should not modify the base or toolkit Dockers directly, if you need you
  >   should create a custom Docker while inheriting the Docker you wish to modify
  
example uses:
  Build a custom docker:                    python custom-tools.py build bash/v1.0
  Build the base and toolkit Dockers:       python custom-tools.py build
  Build a custom docker without cache:      python custom-tools.py build -nc bash/v1.0
  ''')
        
        parser.add_argument('target', help='The target docker with tag (image/tag)', nargs='*', type=str)
        parser.add_argument('-nc',    help='does not use the Docker cache (default - use)',     required=False, default=False, action='store_true')
        parser.add_argument('-f',     help='force the build without prompts (default - false)', required=False, default=False, action='store_true')
        args = parser.parse_args(sys.argv[2:])
        
        build_dockers_path = []
        build_dockers_names = []
        
        if len(args.target) <= 0:
            # Get the list of base and toolkit dockers
            for dirname, dirnames, filenames in os.walk(self.script_dir + '/registry-jobs'):
                for subdirname in dirnames:
                    full_path = os.path.join(dirname, subdirname)
                    docker_name = philly_reg + dirname.split('registry-')[-1] + ':' + subdirname
                    if 'jobs/custom' not in full_path and os.path.isfile(full_path + '/Dockerfile'):
                        build_dockers_path.append(full_path)
                        build_dockers_names.append(docker_name)
        else:
            # Test that the given targets are valid
            for dockerpath in args.target:
                full_path = self.script_dir + '/registry-jobs/custom/' + dockerpath
                docker_name = philly_reg + 'jobs/custom/' + dockerpath.replace('/', ':')
                if os.path.isfile(full_path + '/Dockerfile'):
                    build_dockers_path.append(full_path)
                    build_dockers_names.append(docker_name)
                else:
                    print('WARNING: Cannot find the Dockerfile in ' + full_path)
        
        # Error if no dockers to be built
        if len(build_dockers_path) <= 0:
            print('ERROR: No dockers found to build, aborting...')
            exit(1)
        
        # Print warning
        if not args.f:
            print('\n==============================================================================')
            print('The following dockers will be built:\n')
            for docker_name in build_dockers_names:
                print('    ' + docker_name)
            print('\nHit Ctrl-C to stop, return to proceed.')
            print('==============================================================================\n')
            sys.stdin.read(1)
        
        # Build all the docker images
        for docker_path, docker_name in zip(build_dockers_path, build_dockers_names):
            no_cache = '' if not args.nc else '--no-cache'
            if command_wrapper(['docker', 'build', no_cache, '-t', docker_name, docker_path]) != 'successful':
                exit(1)
    
    def test(self):
        """ Test the target custom docker """
        # Parse the command line
        parser = argparse.ArgumentParser(
            description='Test the target custom docker for requirements',
            usage='''custom-tools.py test [-h] target''',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''
notes:
  > \'target\' is the directory in the \'custom\' directory (registry-jobs/custom/)
  >   Example: bash/v1.0
  > If no config file or data directory provided the script will try to auto discover them
  >   The configuration file and data directory should be placed in the docker folder
  
example uses:
  Test a Custom Docker:                 python custom-tools.py test bash/v1.0
  ''')
        
        parser.add_argument('target', help='The target docker with tag (image/tag)', type=str)
        args = parser.parse_args(sys.argv[2:])
        
        # Local variables
        required = ['Dockerfile', 'toolkit-execute', 'jenkins_config']
        
        # Get the full path of the custom Docker
        dockerdir = self.script_dir + '/registry-jobs/custom/' + args.target
        print('\n************** Testing jobs/custom/' + args.target + ' **************')
        
        # Check for the docker directory
        if os.path.isdir(dockerdir):
            print('PASSED: Docker directory exists (' + dockerdir + ')')
        else:
            print('ERROR: Cannot find the docker directory ' + dockerdir)
            exit(1)
        
        # Declare variables
        test_for_toolkit_execute = False
        
        # Test for the directories and files for the smoke test
        executable = find_executable(dockerdir)
        if not executable:
            print('\nERROR: No "executable*" file found in ' + dockerdir)
            print('       This is needed for the smoke test!')
            exit(1)
        if not os.path.isdir(os.path.join(dockerdir, 'executableData')):
            print('\nERROR: No executableData directory found in ' + dockerdir)
            print('       This is needed for the smoke test!')
            exit(1)
        if os.listdir(os.path.join(dockerdir, 'executableData')) == []:
            print('\nERROR: The executableData directory empty in ' + dockerdir)
            print('       Data is needed for the smoke test!')
            exit(1)
        print('    Found directories and files for the smoke test.')
        
        # Get the inheritied docker
        fromline = get_from_line(os.path.join(dockerdir, 'Dockerfile'))
        inherit_docker = fromline.split()[1]
        
        # If not inheriting from toolkit, toolkit-execute must be present
        if not inherit_docker.startswith('phillyreg.azurecr.io/philly/jobs/toolkit'):
            print('    WARNING: You are not inheriting from a toolkit docker, extra requirements must be met...')
            test_for_toolkit_execute = True
        
        # Test for required LABELS
        required_labels_regex = {}
        required_labels_regex['description'] = r'.+'
        required_labels_regex['repository'] = r'philly/jobs/(custom|toolkit)/.+'
        required_labels_regex['tag'] = r'.+'
        required_labels_regex['creator'] = r'.+'
        required_labels_regex['tooltype'] = r'[a-z]+'
        required_labels_regex['created'] = r'\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}'
        
        labels = get_labels(os.path.join(dockerdir, 'Dockerfile'))
        for label in labels:
            if label[0] not in required_labels_regex:
                print('    The following label not required, ignoring -- ' + label[0])
            elif re.match(required_labels_regex[label[0]], label[1]):
                print('    Found the label ' + label[0] + ' which equals "' + label[1] + '"')
            else:
                print('\nERROR: Label ' + label[0] + ' did not matach regex: ' + required_labels_regex[label[0]])
                exit(1)
       
        # Testing for toolkit-execute
        if test_for_toolkit_execute:
        # Test if the file is a bash file
            # Test for the toolkit-execute bash script
            if not line_exists_in_file('COPY toolkit-execute /home/job/toolkit-execute', os.path.join(dockerdir, 'Dockerfile')):
                print('\nERROR: No "COPY toolkit-execute /home/job/toolkit-execute" found in ' + os.path.join(dockerdir, 'Dockerfile'))
                exit(1)
            else:
                print('    Found "COPY toolkit-execute /home/job/toolkit-execute" found in ' + os.path.join(dockerdir, 'Dockerfile'))
                
            # Test for the toolkit-execute bash script
            if not os.path.isfile(os.path.join(dockerdir, 'toolkit-execute')):
                print('\nERROR: No "toolkit-execute" file found in ' + dockerdir)
                exit(1)
            else:
                print('    Found toolkit-execute file in ' + dockerdir)
            
            # Test that #!/bin/bash is the first dockerdir
            line, number = find_line(os.path.join(dockerdir, 'toolkit-execute'), '^#!/bin/bash$')
            if line is None:
                print('ERROR: Cannot find \'#!/bin/bash\' in toolkit-execute')
                exit(1)
            elif number != 0:
                print('ERROR: \'#!/bin/bash\' is not the first line in toolkit-execute')
                exit(1)
            else:
                print('    Found \'#!/bin/bash\' in toolkit-execute as first line')
                
            # Test that the file is executable
            if not os.access(os.path.join(dockerdir, 'toolkit-execute'), os.X_OK):
                print('\nERROR: "toolkit-execute" file is not executable')
                exit(1)
            else:
                print('    Found that toolkit-execute is executable')
        
        # Test the inheritance
        from_line, number = find_line(dockerdir + '/Dockerfile', '^FROM')
        if 'phillyregistry.azurecr.io/philly/jobs/toolkit/' not in from_line.rstrip('\n'):
            print('\n**************************************************************************')
            print('   WARNING: You are not inheriting from a phillyregistry.azurecr.io image,')
            print('            Philly support will be limited for this docker container.')
            print('**************************************************************************')
        
        # Print success
        print('\nCustom docker passed tests, please check for warnings')
    
    def run(self):
        """ Execute a local test of the docker container """
        # Parse the command line
        parser = argparse.ArgumentParser(
            description='Execute a local test of the docker container',
            usage='''custom-tools.py run [-h] target [EXTRA_ARGS ...]''',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''
notes:
  > \'target\' is the directory in the \'custom\' directory (registry-jobs/custom/)
  >   Example: bash/v1.0
  > EXTRA_ARGS will be passed to the execution of toolkit-execute
  
example uses:
  Run a Custom Docker:                  python custom-tools.py run bash/v1.0
  Run a Custom Docker with EXTRA_ARGS:  python custom-tools.py run bash/v1.0 -e extra
  ''')
        
        parser.add_argument('target', help='The target docker with tag (image/tag)', nargs='+', type=str)
        args = parser.parse_args(sys.argv[2:])
        
        full_path = self.script_dir + '/registry-jobs/custom/' + args.target[0]
        docker_name = 'phillyregistry.azurecr.io/philly/jobs/custom/' + args.target[0].replace('/', ':')
        print('\n************** Running ' + docker_name + ' **************\n')
        
        # Read in the configuration file and data directory
        config_file = find_executable(full_path)
        data_dir = 'executableData/'
        
        # Construct other variables for the test run
        output_dir = self.script_dir + '/run-output/' + args.target[0]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print('Configuration file: ' + config_file)
        print('Data Directory: ' + data_dir)
        print('Output Directory: ' + output_dir)
        
        # Create and execute the docker command
        docker_command = ['nvidia-docker', 'run', '-ti', '--rm', '-v', full_path + ':/var/scratch', '-v', full_path + ':/var/hdfs', '-v', output_dir + ':/var/logs', '-v', output_dir + ':/var/models', docker_name]
        docker_command.extend(['/home/job/toolkit-execute', '--configFile', '/var/scratch/' + config_file, '--dataDir', '/var/hdfs/' + data_dir, '--logDir', '/var/logs', '--modelDir', '/var/models', ' '.join(args.target[1:])])
        command_wrapper(docker_command)

if __name__ == "__main__":
    """ Kick off the script """
    Tools()