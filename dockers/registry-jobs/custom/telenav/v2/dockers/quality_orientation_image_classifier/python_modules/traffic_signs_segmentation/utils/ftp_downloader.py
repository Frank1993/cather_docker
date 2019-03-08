"""
Download tool for resources on ftp
"""

import argparse

import apollo_python_common.ftp_utils as ftp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ftp_file_path",
                        type=str, required=True)
    parser.add_argument("-o", "--local_file_path",
                        type=str, required=True)
    parser.add_argument("-f", "--ftp_server",
                        type=str, required=False, default="10.230.2.19")
    parser.add_argument("-u", "--ftp_username",
                        type=str, required=False, default="orbb_ftp")
    parser.add_argument("-p", "--ftp_password",
                        type=str, required=False, default="plm^2")
    args = parser.parse_args()
    ftp.file_copy_ftp_to_local( args.ftp_server, args.ftp_username, args.ftp_password, args.ftp_file_path, args.local_file_path )

if __name__ == "__main__":
    main()