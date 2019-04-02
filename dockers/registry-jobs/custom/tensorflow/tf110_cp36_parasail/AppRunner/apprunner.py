import logging
import os
import sys

import app_launcher
import runtime_builder
import runtime_installer
import runtime_loader
from shared import util, consts, argument_parser


def parse_argument(argv):
    parser = argument_parser.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subparser_name')

    # run
    subparser = subparsers.add_parser(consts.CMD_RUN, help='run app')
    subparser.add_argument(consts.ARG_HOST_PLATFORM, required=True)
    subparser.add_argument(consts.ARG_APP_TYPE, required=True)
    subparser.add_argument(consts.ARG_RUNTIME_BINARY, nargs='?')
    subparser.add_argument(consts.ARG_RUNTIME_NAME)
    subparser.add_argument(consts.ARG_RUNTIME_VERSION)
    subparser.add_argument(consts.ARG_RUNTIME_SOURCE)
    subparser.add_argument(consts.ARG_APP_ARGS, nargs='?')
    subparser.add_argument(consts.ARG_SETUP_ONLY, action="store_true", default=False)
    subparser.add_argument(consts.ARG_APPRUNNER_WORKING_ROOT)
    subparser.add_argument(consts.ARG_APPRUNNER_DATA_ROOT)
    subparser.set_defaults(func=run)

    # load
    subparser = subparsers.add_parser(consts.CMD_LOAD_RUNTIME, help="load runtime")
    subparser.add_argument(consts.ARG_HOST_PLATFORM, required=True)
    subparser.add_argument(consts.ARG_APP_TYPE, required=True)
    subparser.add_argument(consts.ARG_RUNTIME_BINARY, nargs='?')
    subparser.add_argument(consts.ARG_RUNTIME_NAME)
    subparser.add_argument(consts.ARG_RUNTIME_VERSION)
    subparser.add_argument(consts.ARG_RUNTIME_SOURCE)
    subparser.add_argument(consts.ARG_APPRUNNER_WORKING_ROOT)
    subparser.add_argument(consts.ARG_APPRUNNER_DATA_ROOT)
    subparser.set_defaults(func=load_runtime)

    # install
    subparser = subparsers.add_parser(consts.CMD_INSTALL_RUNTIME, help="install runtime")
    subparser.add_argument(consts.ARG_HOST_PLATFORM, required=True)
    subparser.add_argument(consts.ARG_INSTALL_SOURCE, required=True)
    subparser.add_argument(consts.ARG_INSTALL_TARGET, required=True)
    subparser.set_defaults(func=install_runtime)

    # launch
    subparser = subparsers.add_parser(consts.CMD_LAUNCH_APP, help="launch app")
    subparser.add_argument(consts.ARG_HOST_PLATFORM, required=True)
    subparser.add_argument(consts.ARG_APP_TYPE, required=True)
    subparser.add_argument(consts.ARG_APP_ARGS, nargs='?')
    subparser.add_argument(consts.ARG_APPRUNNER_WORKING_ROOT)
    subparser.set_defaults(func=launch_app)

    # build
    subparser = subparsers.add_parser(consts.CMD_BUILD_RUNTIME, help="build runtime")
    subparser.add_argument(consts.ARG_BUILD_SOURCE, required=True)
    subparser.add_argument(consts.ARG_ARCHITECTURES)
    subparser.add_argument(consts.ARG_PUBLISH_DIR)
    subparser.add_argument(consts.ARG_LOCAL_PACKAGE_DIR)
    subparser.add_argument(consts.ARG_USE_TEMP_BUILD_DIR, action="store_true", default=False)
    subparser.add_argument(consts.ARG_RESERVE_BUILD_DIR, action="store_true", default=False)
    subparser.add_argument(consts.ARG_COPY_SITE_PACKAGES, action="store_true", default=False)
    subparser.add_argument(consts.ARG_APPRUNNER_DATA_ROOT)
    subparser.set_defaults(func=build_runtime)

    if len(sys.argv) <= 2:
        parser.print_help()
        util.abort()

    (args, unknown) = parser.parse_known_args(argv)
    return args, unknown


def execute_cmd(cmd, argv=None):
    (args, unknown) = parse_argument([cmd] + argv)
    util.print_cmd_args(args)
    args.func(args, argv)


def run(args, argv=None):
    load_runtime(args, argv)
    invoke_install(args, argv)

    setup_only = util.get_arg_value(args, consts.ARG_SETUP_ONLY, False)
    if not setup_only:
        launch_app(args, argv)


def load_runtime(args, argv=None):
    if not runtime_loader.run(args):
        util.abort("Load runtime failed")


def invoke_install(args, argv):
    host_platform = util.get_arg_value(args, consts.ARG_HOST_PLATFORM)
    working_root = util.get_working_root(args)
    runtime_staging_dir = os.path.join(working_root, consts.RUNTIME_STAGING_DIR_NAME)
    runtime_dir = os.path.join(working_root, consts.RUNTIME_DIR_NAME)

    util.recreate_directory(runtime_dir)

    script_path = os.path.join(runtime_staging_dir, consts.RUNTIME_INSTALL_SCRIPT_NAME)
    if not os.path.isfile(script_path):
        logging.info("No install script found, perform copy")
        util.copy_directory(runtime_staging_dir, runtime_dir)
    else:
        script_args = " ".join([host_platform, runtime_dir])
        cmd = " ".join([util.get_current_python_path(), script_path, script_args])

        logging.info("Invoke runtime install script: %s", cmd)
        status = os.system(cmd)
        if status == 0:
            logging.info("Invoke install succeed")
        else:
            util.abort("Invoke install failed")


def install_runtime(args, argv=None):
    if not runtime_installer.run(args):
        util.abort("Install runtime failed")


def launch_app(args, argv=None):
    if not app_launcher.run(args, argv):
        util.abort("Launch app failed")


def build_runtime(args, argv=None):
    if not runtime_builder.run(args):
        util.abort("Build runtime failed")


logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)

if __name__ == '__main__':
    (args, unknown) = parse_argument(sys.argv[1:])
    util.print_argv()
    util.print_cmd_args(args)

    if util.get_python_version_info().major < 3:
        util.abort("Python3 is required")

    args.func(args, sys.argv[2:])
