import ftplib
import os
import shutil
import zipfile

import apollo_python_common.io_utils as io_utils

FTP_SERVER = '10.230.2.19'
FTP_USER_NAME = 'orbb_ftp'
FTP_PASSWORD = 'plm^2'


def file_copy_local_to_ftp(ftp_ip, ftp_login, ftp_passwd, local_file_path, ftp_file_path):
    ftp_folder = '/'.join(ftp_file_path.split('/')[:-1])
    ftp_filename = ftp_file_path.split('/')[-1]
    ftp = ftplib.FTP(ftp_ip, ftp_login, ftp_passwd)
    try:
        ftp.cwd(ftp_folder)
    except:
        ftp.mkd(ftp_folder)
        ftp.cwd(ftp_folder)
    with open(local_file_path, 'rb') as f:
        ftp.storbinary('STOR ' + ftp_filename, f)
    ftp.quit()


def file_copy_ftp_to_local(ftp_ip, ftp_login, ftp_passwd, ftp_file_path, local_file_path):
    if os.path.dirname(local_file_path) != "":
        io_utils.create_folder(os.path.dirname(local_file_path))
    ftp_folder = os.path.dirname(ftp_file_path)
    ftp_filename = os.path.basename(ftp_file_path)
    ftp = ftplib.FTP(ftp_ip, ftp_login, ftp_passwd)
    ftp.cwd(ftp_folder)
    with open(local_file_path, 'wb') as f:
        ftp.retrbinary('RETR ' + ftp_filename, f.write)


def dir_copy_local_to_ftp(ftp_ip, ftp_login, ftp_passwd, local_dir_path, ftp_file_path):
    temp_zip_path = "./temp_zip"
    shutil.make_archive(temp_zip_path, 'zip', local_dir_path)
    file_copy_local_to_ftp(ftp_ip, ftp_login, ftp_passwd, temp_zip_path + ".zip", ftp_file_path)
    os.remove(temp_zip_path + ".zip")


def copy_zip_and_extract(ftp_bundle_path, local_bundle_path):
    local_zip_path = './ftp_bundle.zip'
    file_copy_ftp_to_local(FTP_SERVER, FTP_USER_NAME, FTP_PASSWORD,
                           ftp_bundle_path,
                           local_zip_path)

    with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
        zip_ref.extractall(local_bundle_path)

    os.remove(local_zip_path)
