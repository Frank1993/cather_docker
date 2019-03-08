import os
import zipfile

import apollo_python_common.ftp_utils as ftp_utils

LOCAL_TEST_RESOURCES_FOLDER = "/usr/local/share/ORBB/resources/python_tests"


def copy_test_file_from_ftp_to_local(ftp_path, local_relative_folder):
    '''
    Copy specified resource file from FTP to the local test resources folder.
    If the specified file already exists in the local path it will be overwritten.
    :param file_ftp_relative_path: the path in the FTP tests resources folder
    :param local_relative_folder: the path in the local tests resources folder
    :return: file's local full path
    '''
    file_name = os.path.basename(ftp_path)
    local_path = os.path.join(LOCAL_TEST_RESOURCES_FOLDER, local_relative_folder, file_name)
    if os.path.isfile(local_path):
        os.remove(local_path)
    ftp_utils.file_copy_ftp_to_local(ftp_utils.FTP_SERVER, ftp_utils.FTP_USER_NAME, ftp_utils.FTP_PASSWORD,
                                     ftp_path, local_path)
    return local_path


def ensure_test_resource(ftp_test_path, local_path):
    '''
    Ensures test resources for roi_ssd tests.
    :return: The local path where resources were stored.
    '''
    full_local_path = copy_test_file_from_ftp_to_local(ftp_test_path, local_path)
    with zipfile.ZipFile(full_local_path, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(full_local_path))
    os.remove(full_local_path)
    return os.path.dirname(full_local_path)

