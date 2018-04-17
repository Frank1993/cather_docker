import os
import re
import argparse

from git import Repo
from pprint import pprint

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

# Find if a line exists in a file
def line_exists_in_file(search_line, filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    for line in lines:
        if line == search_line:
            return True
    return False

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

def save_dockers(dockers, filename):
    if len(dockers) > 0:
        with open(filename, 'w') as f:
            f.write(dockers)

# Find Modified Dockers
def find_modified_dockers(repo, source_branch, compare_target):
    dockers = []
    print 'Getting common ancestor of', source_branch, 'and', compare_target
    common_ancestor = str(repo.merge_base(source_branch, compare_target)[0])
    print 'Finding modified docker using diff between', common_ancestor, 'and', source_branch
    response = repo.git.diff('--name-status', common_ancestor, source_branch)
    print('\nResponse to git diff:\n' + response)

    for line in response.splitlines():
        split = line.split()
        split[1] = os.path.dirname(split[1])
        if split[1].startswith('dockers/registry-jobs/') and os.path.isfile(os.path.join(split[1], 'Dockerfile')) and split not in dockers:
            dockers.append(split)

    return dockers

# Sort Modified Dockers
# This method will sort into two lists:
#   deleted_dockers - dockers to delete
#   modified_dockers - dockers that need to be built
# The modified_dockers will be sorted for the following inheritance:
#   docker.io <-- base <-- toolkit <-- custom
#   where any docker can inherit from one bellow but not from
#     a higher or same docker layer
def sort_modified_dockers(dockers):
    modified_dockers = []
    deleted_dockers = []

    # Find the deleted dockers
    templist = []
    for docker in dockers:
        if docker[0] == 'D':
            deleted_dockers.append(docker[1])
        else:
            templist.append(docker)
    dockers = templist

    # Add the dockers from base directory
    subdirs = ['base', 'toolkit', 'custom']
    for subdir in subdirs:
        templist = []
        for docker in dockers:
            if docker[1].startswith('dockers/registry-jobs/' + subdir):
                modified_dockers.append(docker[1])
            else:
                templist.append(docker)
        dockers = templist

    return modified_dockers, deleted_dockers

def inherited_docker_modified(dockerdir, dockers):
    fromline = get_from_line(os.path.join(dockerdir, 'Dockerfile'))
    inherited_docker = fromline.split()[1]
    if inherited_docker.startswith('phillyregistry'):
        inherited_directory = inherited_docker.replace('phillyregistry.azurecr.io/philly/', 'dockers/registry-').replace(':', '/')
        if inherited_directory in dockers:
            return inherited_directory
        return inherited_docker_modified(inherited_directory, dockers)
    return None

# Test that the modified dockers conform to expected formats
def test_modified_dockers(dockers):
    for dockerdir in dockers:
        print('\nTesting Docker: ' + dockerdir)

        # Declare variables
        test_for_toolkit_execute = False

        # No special tests needed for base dockers
        if dockerdir.startswith('dockers/registry-jobs/base/'):
            print('    No special tests needed for base dockers.')
            continue

        # Tests for toolkit dockers only
        if dockerdir.startswith('dockers/registry-jobs/toolkit/'):
            test_for_toolkit_execute = True

        # Tests for custom dockers only
        if dockerdir.startswith('dockers/registry-jobs/custom/'):
            # Test for the directories and files for the smoke test
            # For now these files are not needed so commenting out this code
            # If smoke testing in CCP is brought back then these lines need to be uncommented
            #executable = find_executable(dockerdir)
            #if not executable:
                #print('\nERROR: No "executable*" file found in ' + dockerdir)
                #print('       This is needed for the smoke test within CCP cluster!')
                #exit(1)
            #if not os.path.isdir(os.path.join(dockerdir, 'executableData')):
                #print('\nERROR: No executableData directory found in ' + dockerdir)
                #print('       This is needed for the smoke test within CCP cluster!')
                #exit(1)
            #if os.listdir(os.path.join(dockerdir, 'executableData')) == []:
                #print('\nERROR: The executableData directory empty in ' + dockerdir)
                #print('       Data is needed for the smoke test within CCP cluster!')
                #exit(1)
            #print('    Found directories and files for the smoke test within CCP.')

            # Get the inheritied docker
            fromline = get_from_line(os.path.join(dockerdir, 'Dockerfile'))
            inherit_docker = fromline.split()[1]

            # If not inheriting from toolkit, toolkit-execute must be present
            if not inherit_docker.startswith('phillyreg.azurecr.io/philly/jobs/toolkit'):
                print('    WARNING: You are not inheriting from a toolkit docker, extra requirements must be met...')
                test_for_toolkit_execute = True

        # The following must be tested for all docker that are not base

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

            # Test that the file is executable
            if not os.access(os.path.join(dockerdir, 'toolkit-execute'), os.X_OK):
                print('\nERROR: "toolkit-execute" file is not executable')
                exit(1)
            else:
                print('    Found that toolkit-execute is executable')

        # Test the inheritance of the docker.  A inherited docker cannot be built
        # at the same time has the child docker.  The script currently defaults to pull
        # the inherited docker to the local registry.  If demand is high for building
        # inherited images, then we need to add more detailed logic to BuildDockers
        inherited_docker = inherited_docker_modified(dockerdir, dockers)
        if inherited_docker is not None:
            print('\nERROR: Building an inherited docker: ' + inherited_docker)
            print('       At this time, it is not valid to modify an inherited docker and')
            print('       a child docker in the same PR.  Please create and finish a PR for')
            print('       the inherited docker before modifing:')
            print('       ' + dockerdir)
            exit(2)

    return True

# main: where all the magic happens
if __name__ == "__main__":

    # parse the command line
    parser = argparse.ArgumentParser(description='Discover the dockers that have been modified', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', help='action type for this script (build|publish)', default='build')
    parser.add_argument('-t', '--test', help='Runs a test iteration', default=False, required=False, action='store_true')
    args = parser.parse_args()

    # Check that the repository is not empty
    repo = Repo(os.getcwd())
    assert not repo.bare

    if args.test:
        os.environ['BUILD_REQUESTEDFOR'] = "test"
        os.environ['BUILD_REQUESTEDFOREMAIL'] = "test_script@microsoft.com"
        os.environ['AGENT_TEMPDIRECTORY'] = "/tmp"
        os.environ['BUILD_BUILDNUMBER'] = "1"
        os.environ["SYSTEM_PULLREQUEST_SOURCEBRANCH"] = "refs/heads/" + repo.head.ref.name
        os.environ['SYSTEM_PULLREQUEST_TARGETBRANCH'] = "master"

    # Get environment variables
    user = os.environ['BUILD_REQUESTEDFOR']
    user_email = os.environ['BUILD_REQUESTEDFOREMAIL']
    temp_directory = os.environ['AGENT_TEMPDIRECTORY']
    build_number = os.environ['BUILD_BUILDNUMBER']


    # Print the environment variables
    print('\nBuild Information:')
    print('    User = ' + user)
    print('    User Email = ' + user_email)
    print('    Temp directory = ' + temp_directory)
    print('    Build Number = ' + build_number)

    if args.action == 'build':
        try:
            source_branch = os.environ['SYSTEM_PULLREQUEST_SOURCEBRANCH'].replace('refs/heads/', '')
            compare_target = os.environ['SYSTEM_PULLREQUEST_TARGETBRANCH'].replace('refs/heads/', '')
            print('    Source Branch = ' + source_branch)
            print('    Target Branch = ' + compare_target)
        except:
            source_branch = os.environ['BUILD_SOURCEBRANCH'].replace('refs/heads/', '')
            compare_target = 'master'
            print('    Source Branch = ' + source_branch)
            print('    Target Branch = ' + compare_target)
        if compare_target != 'master':
            print('\nPR is not for master, no work is needed...')
            exit(0)
        source_branch = 'remotes/origin/' + source_branch
        compare_target = 'remotes/origin/' + compare_target
    elif args.action == 'publish':
        # We trigger this either via continuous integration, in which case we rely on all changes to be in the last commit (squash merge
        # on master), or manual in which case we take a diff with master.
        reason = os.getenv('BUILD_REASON', None)
        if reason == 'IndividualCI':
            source_branch = 'HEAD'    # Here we assume a squash merge, so we would have all the changes in the last commit.
            compare_target = 'HEAD~1'
        elif reason == 'Manual':
            source_branch = os.getenv('BUILD_SOURCEVERSION')
            print source_branch
            if not source_branch:
                raise Exception('ERROR: Couldn\'t find a source version to compare against.')
                exit(1)
            compare_target = 'master'
        else:
            raise Exception('Error: Unsupported BUILD_REASON "' + reason + '".')
    else:
        print('ERROR: Improper action (action = ' + args.action + ')')
        exit(1)

    # Find the dockers that were modified
    dockers = find_modified_dockers(repo, source_branch, compare_target)
    modified_dockers, deleted_dockers = sort_modified_dockers(dockers)

    # Print the found dockers
    print('\nDeleted Dockers:')
    pprint(deleted_dockers)
    print('\nModified Dockers:')
    pprint(modified_dockers)

    if args.action == 'build':
        # Test the modified dockers
        test_modified_dockers(modified_dockers)

    # Add the dockers to the environment variables
    modified = deleted = ""
    for docker in modified_dockers:
        modified += docker + ";"
    for docker in deleted_dockers:
        deleted += docker + ";"
    save_dockers(modified[:-1], temp_directory + '/modified_dockers')
    save_dockers(deleted[:-1], temp_directory + '/deleted_dockers')

    print('\nSuccessfully found modified/deleted docker(s)')
    exit(0)
