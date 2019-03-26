import argparse
import copy
import logging
import os
import re
import signal
import stat
import subprocess
import tempfile
import time
import traceback
from collections import namedtuple

from shared import consts, util


class NodeInfo:
    def __init__(self, id: int, local_id: int, name, ip, ports, gpus=None):
        self.id = id
        self.local_id = local_id
        self.name = name
        self.full_name = "Node_%s_%s" % (id, name)
        self.ip = ip
        self.ports = list(map(str, ports))
        if gpus is None:
            self.gpus = []
        else:
            self.gpus = list(map(str, gpus))

    def is_master(self):
        return self.id == 0

    def is_local_master(self):
        return self.local_id == 0


class LaunchSetting:
    def __init__(self, name, cmd, envs=None):
        self.name = name
        self.cmd = cmd
        if envs is None:
            self.envs = []
        else:
            self.envs = envs


class JobSetting:
    def __init__(self, name, task_count=0, gpu_count=0, port_count=0):
        self.name = name
        self.task_count = task_count
        self.gpu_count = gpu_count
        self.port_count = port_count


TaskSetting = namedtuple('TaskSetting', 'node job_name task_name task_index port gpus')


class TensorFlowLauncher:
    APP_INPUT_ARG_PREFIX = "--input-"
    APP_OUTPUT_ARG_PREFIX = "--output-"
    ARG_APP_SCRIPT = "--app-script"
    ARG_INPUT_APP_DIRECTORY = "--input-app-dir"
    ARG_LOG_DIRECTORY = "--log-dir"
    ARG_EXE_MODE = "--exe-mode"
    ARG_JOB_CONFIG = "--job-config"
    ARG_PIP_PACKAGES = "--pip-packages"
    ARG_OFFLINE_PACKAGE_ONLY = "--offline"
    ARG_AUTO_KILL_PS = "--auto-kill-ps"
    ARG_LOCAL_MODE = "--local-mode"
    ARG_SLEEP_AFTER_FINISHED_IN_SECONDS = "--sleep"
    ARG_USE_UNDERSCORE_ARG = "--use-underscore-arg"

    EXE_MODE_SINGLETON = "Singleton"
    EXE_MODE_PS_WORKER = "PS-Worker"
    EXE_MODE_MPI = "MPI"
    EXE_MODE_Custom = "Custom"

    ARG_NODE_COUNT = "--node-count"
    ARG_NODE_LIST_PATH = "--node-list-path"
    ARG_NODE_ID = "--node-id"
    ARG_NODE_NAME = "--node-name"

    ARG_TF_JOB_NAME = "--job-name"
    ARG_TF_TASK_INDEX = "--task-index"
    ARG_TF_JOB_SPEC = "--%s-hosts"

    def __init__(self, args, argv):
        self.args = args
        self.argv = argv

    def run(self):
        try:
            self.init()
            self.load_node_list()
            self.setup()
            self.generate_launch_settings()
            return self.launch()
        except:
            logging.error(traceback.format_exc())
            return False

    def init(self):
        self.os_name = util.get_os_name()
        self.ip = util.get_host_ip()
        self.host_platform = self.get_arg_value(consts.ARG_HOST_PLATFORM)
        self.working_root = util.get_working_root(self.args)
        self.runtime_dir = os.path.join(util.get_working_root(self.args), consts.RUNTIME_DIR_NAME)
        self.python_dir = os.path.join(self.runtime_dir, "python")
        self.env_dir = os.path.join(self.runtime_dir, "env")
        self.node_list_path = os.path.join(self.runtime_dir, "node_list.txt")

        # Use python executable under env folder
        self.python_path = util.locate_python3_exe(self.env_dir, self.os_name)
        # For backward compatibility, previous runtime image does not have an env directory
        if not os.path.isdir(self.env_dir):
            self.python_path = util.locate_python3_exe(self.python_dir, self.os_name)

        self.app_dir = os.path.join(self.runtime_dir, "app")
        self.app_args_str = self.get_arg_value(consts.ARG_APP_ARGS, "")

        parser = argparse.ArgumentParser()
        parser.add_argument(self.ARG_LOG_DIRECTORY)
        parser.add_argument(self.ARG_EXE_MODE)
        parser.add_argument(self.ARG_JOB_CONFIG, nargs='?')
        parser.add_argument(self.ARG_APP_SCRIPT, required=True)
        parser.add_argument(self.ARG_PIP_PACKAGES)
        parser.add_argument(self.ARG_OFFLINE_PACKAGE_ONLY, action="store_true", default=False)
        parser.add_argument(self.ARG_AUTO_KILL_PS, action="store_true", default=False)
        parser.add_argument(self.ARG_LOCAL_MODE, action="store_true", default=False)
        parser.add_argument(self.ARG_SLEEP_AFTER_FINISHED_IN_SECONDS, nargs='?', const="3600")
        parser.add_argument(self.ARG_USE_UNDERSCORE_ARG, action="store_true", default=False)

        input_arg_names = self.extract_io_arg_names(self.argv, self.APP_INPUT_ARG_PREFIX)
        output_arg_names = self.extract_io_arg_names(self.argv, self.APP_OUTPUT_ARG_PREFIX)
        for arg_name in (input_arg_names + output_arg_names):
            parser.add_argument(arg_name, nargs='?')

        (args, unknown) = parser.parse_known_args(self.argv)

        self.input_app_dir = util.get_arg_value(args, self.ARG_INPUT_APP_DIRECTORY)
        if not self.input_app_dir:
            raise RuntimeError("Input app directory is not set")

        app_script = util.get_arg_value(args, self.ARG_APP_SCRIPT)
        if not app_script:
            raise RuntimeError("App script is not set")

        self.app_script = os.path.join(self.app_dir, app_script)
        logging.info("App script path: %s", self.app_script)

        self.input_args = []
        for arg_name in input_arg_names:
            if arg_name != self.ARG_INPUT_APP_DIRECTORY:
                self.input_args.append((arg_name, util.get_arg_value(args, arg_name)))

        self.output_args = []
        for arg_name in output_arg_names:
            self.output_args.append((arg_name, util.get_arg_value(args, arg_name)))

        self.log_dir = util.get_arg_value(args, self.ARG_LOG_DIRECTORY)
        if self.log_dir:
            self.log_dir = os.path.abspath(self.log_dir)

        self.print_io_paths()

        self.exe_mode = util.get_arg_value(args, self.ARG_EXE_MODE, self.EXE_MODE_SINGLETON)
        self.job_config = util.get_arg_value(args, self.ARG_JOB_CONFIG)
        self.app_status_refresh_interval_in_seconds = 10
        self.pip_packages = util.get_arg_value(args, self.ARG_PIP_PACKAGES)
        self.is_offline_package_only = util.get_arg_value(args, self.ARG_OFFLINE_PACKAGE_ONLY)
        self.is_local_mode = util.get_arg_value(args, self.ARG_LOCAL_MODE)
        if self.is_local_mode:
            logging.info("!!!!!!!!!!!! Running in local mode !!!!!!!!!!!!!!!")
        self.sleep_after_finished_in_seconds = int(
            util.get_arg_value(args, self.ARG_SLEEP_AFTER_FINISHED_IN_SECONDS, "0"))
        self.use_underscore_arg = util.get_arg_value(args, self.ARG_USE_UNDERSCORE_ARG)

        # total worker count, used only when exe mode is PS-Worker
        self.total_worker_count = 0
        if self.exe_mode == self.EXE_MODE_PS_WORKER:
            self.auto_kill_ps = util.get_arg_value(args, self.ARG_AUTO_KILL_PS)
        else:
            self.auto_kill_ps = False

    def extract_io_arg_names(self, argv, prefix):
        arg_names = []
        for arg in argv:
            if arg.startswith(prefix):
                arg_names.append(arg)
        return arg_names

    def print_io_paths(self):
        logging.info("Input paths:")
        for arg in self.input_args:
            logging.info("\t%s=%s", arg[0], arg[1])

        logging.info("Output paths:")
        for arg in self.output_args:
            logging.info("\t%s=%s", arg[0], arg[1])

        if self.log_dir:
            logging.info("\t%s=%s", self.ARG_LOG_DIRECTORY, self.log_dir)

    def load_node_list(self):
        node_list, current_node_id = self.parse_node_list()
        current_node = node_list[current_node_id]
        self.node_list = node_list
        self.node_count = len(node_list)
        self.current_node = current_node
        self.print_node_list(node_list)

    def parse_node_list(self):
        raise NotImplementedError()

    def save_node_list(self, path, nodes):
        with open(path, 'w') as file:
            for node in nodes:
                file.write(
                    '\t'.join([node.name, str(node.id), str(node.local_id), node.ip, ','.join(map(str, node.ports)),
                               ','.join(map(str, node.gpus))]))
                file.write('\n')

    def print_node_list(self, node_list):
        logging.info("Node list:")
        for node in node_list:
            logging.info("\t[%s] %s, local_id=%s, ip=%s, ports=%s, gpus=%s", node.id, node.name, node.local_id, node.ip,
                         ','.join(node.ports),
                         ','.join(node.gpus))

    def setup(self):
        done_path = os.path.join(self.runtime_dir, "setup.done")
        if self.current_node.is_local_master():
            logging.info("Perform setup on %s", self.current_node.full_name)
            self.save_node_list(self.node_list_path, self.node_list)

            input_app_dir = self.input_app_dir
            logging.info("Copy app directory from %s to %s", input_app_dir, self.app_dir)

            if util.is_hdfs_path(input_app_dir):
                util.execute_cmd("hdfs dfs -cp %s %s" % (input_app_dir, self.app_dir))
            elif not os.path.isfile(input_app_dir):
                util.copy_directory(input_app_dir, self.app_dir)
            else:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    zip_path = os.path.join(tmp_dir, "app.zip")
                    util.copy_file(input_app_dir, zip_path)
                    util.unzip(zip_path, self.app_dir)

            self.install_custom_packages()
            util.write_all_text(done_path)
        else:
            logging.info("Wait master apprunner instance to setup")
            if not util.wait_file(done_path):
                raise RuntimeError("Wait master apprunner instance setup timeout")

    def install_custom_packages(self):
        requirements_path = os.path.join(self.app_dir, consts.REQUIREMENTS_TXT)

        if self.pip_packages:
            packages = self.pip_packages.split(',')
            util.append_all_lines(requirements_path, packages)

        if os.path.isfile(requirements_path):
            self.install_pip_packages(requirements_path)

        is_conda_runtime = os.path.isfile(util.locate_conda_exe(self.python_dir, self.os_name))
        if is_conda_runtime:
            conda_spec_path = os.path.join(self.app_dir, consts.CONDA_SPEC_TXT)
            if os.path.isfile(conda_spec_path):
                self.install_conda_packages(conda_spec_path)

            util.resolve_conda_package_conflict(self.python_dir, self.env_dir)

    def install_pip_packages(self, requirements_path):
        logging.info("Install custom pip packages")

        cmd = self.python_path + " -m pip install -r " + requirements_path + " --disable-pip-version-check --no-cache-dir"
        cmd += " --find-links=" + os.path.join(self.app_dir, "packages")
        for package_dir in self.get_offline_pip_package_dirs():
            cmd += " --find-links=" + package_dir

        if self.is_offline_package_only:
            cmd += " --no-index"

        # Re-install packages first to ensure packages are re-installed without changing dependencies
        # Then install packages normally to make sure missing dependencies (if any) are installed
        # This is for scenario like, user provides a custom tensorflow whl and wants to install it.
        #
        # pip -I does not work as expected, it will NOT remove existing package dist-info, which will lead to some problems.
        # E.g. pip freeze reports wrong version, further pip uninstall might break
        # https://github.com/pypa/pip/issues/4400
        cmd_1 = cmd + " -U --force-reinstall --no-deps"
        exit_code, _ = util.execute_cmd(cmd_1, check_exit_code=False)
        if exit_code != 0:
            # On Windows, pip may need to uninstall some packages first before install them
            # Uninstall may fail due to path too long when trying to remove tmp directory
            # Retry install on such failure
            logging.info("Retry install custom pip packages, step 1")
            util.execute_cmd(cmd_1)

        util.execute_cmd(self.python_path + " -m pip freeze --all --disable-pip-version-check")

        exit_code, _ = util.execute_cmd(cmd, check_exit_code=False)
        if exit_code != 0:
            logging.info("Retry install custom pip packages, step 2")
            util.execute_cmd(cmd)

        util.execute_cmd(self.python_path + " -m pip freeze --all --disable-pip-version-check")

    def get_offline_pip_package_dirs(self):
        return []

    def install_conda_packages(self, conda_spec_path):
        logging.info("Install custom conda packages")
        conda_exe = util.locate_conda_exe(self.python_dir, self.os_name)
        cmd = conda_exe + " install -p " + self.env_dir + " -y -q --copy --file=" + conda_spec_path

        if self.is_offline_package_only:
            cmd += " --offline"

        package_dir = os.path.join(self.app_dir, "packages")
        if os.path.isdir(package_dir):
            logging.info("Build conda channel")
            # HACK: assume runtime architecture is same with cpu architecture
            channel_name = util.get_conda_channel_name(self.os_name, util.get_cpu_architecture())
            channel_dir = os.path.join(package_dir, channel_name)
            for package in next(os.walk(package_dir))[2]:
                if package.endswith(".bz2"):
                    util.copy_file(os.path.join(package_dir, package), os.path.join(channel_dir, package))

            util.execute_cmd(conda_exe + " index " + channel_dir)
            cmd = cmd + " -c file:///" + channel_dir

        util.execute_cmd(cmd)

    def generate_launch_settings(self):
        mode = self.exe_mode
        logging.info("Execution mode is %s", mode)

        if mode == self.EXE_MODE_SINGLETON:
            settings = self.generate_singleton_launch_settings()
        elif mode == self.EXE_MODE_PS_WORKER:
            settings = self.generate_ps_worker_launch_settings()
        elif mode == self.EXE_MODE_MPI:
            settings = self.generate_mpi_launch_settings()
        elif mode == self.EXE_MODE_Custom:
            settings = self.generate_custom_launch_settings()
        else:
            raise RuntimeError("Invalid exe mode %s" % mode)

        self.launch_settings = settings

    def generate_singleton_launch_settings(self):
        if self.node_count != 1:
            raise RuntimeError("ExecutionMode is Singleton but node_count is %s" % self.node_count)

        node = self.current_node
        name = node.name
        argv = self.get_common_argv()
        cmd = util.get_normalized_argv(argv) + " " + self.apply_argument_variable(self.app_args_str, argv)
        envs = dict([self.get_cuda_visible_devices_env(node.gpus)])
        return [LaunchSetting(name, cmd, envs)]

    def generate_ps_worker_launch_settings(self):
        job_config = self.job_config
        if not job_config:
            job_config = "ps#worker"

        node_count = self.node_count
        gpu_count_per_node = len(self.current_node.gpus)
        job_settings = self.parse_job_settings(job_config, node_count, gpu_count_per_node)
        self.validate_job_settings(job_settings, node_count)
        self.total_worker_count = job_settings[1].task_count

        # Re-order for even distribution
        node_list = copy.deepcopy(self.node_list)
        ip2nodes = dict()
        for node in node_list:
            if node.ip not in ip2nodes:
                ip2nodes[node.ip] = [node]
            else:
                ip2nodes[node.ip].append(node)
        node_list = list(util.roundrobin(*[kv[1] for kv in sorted(ip2nodes.items(), key=lambda kv: (kv[0]))]))

        task_settings = self.generate_task_settings(job_settings, node_list)
        if self.current_node.is_master():
            self.print_task_settings(task_settings)

        cluster_spec = self.build_cluster_spec(job_settings, task_settings)
        common_argv = self.get_common_argv()
        for (job_name, hosts) in cluster_spec.items():
            common_argv += [self.format_arg_name(self.ARG_TF_JOB_SPEC % job_name), ','.join(hosts)]

        launch_settings = []
        for task in task_settings:
            if task.node.id != self.current_node.id:
                continue
            argv = common_argv + [
                self.format_arg_name(self.ARG_TF_JOB_NAME), task.job_name,
                self.format_arg_name(self.ARG_TF_TASK_INDEX), str(task.task_index)
            ]
            cmd = util.get_normalized_argv(argv) + " " + self.apply_argument_variable(self.app_args_str, argv)
            envs = dict([self.get_cuda_visible_devices_env(task.gpus)])
            launch_settings.append(LaunchSetting(task.task_name, cmd, envs))

        return launch_settings

    def parse_job_settings(self, job_config, node_count, gpu_count_per_node):
        settings = dict()

        for config in job_config.split('#'):
            _ = [s.strip() for s in config.split(',') if s and not s.isspace()]
            kvs = [s.split('=', 1) for s in _]
            job_name = kvs[0][0].strip()

            options = dict()
            for kv in kvs:
                key = kv[0].strip()
                value = None if len(kv) == 1 else kv[1].strip()
                options[key] = value

            task_count = self.parse_job_task_count(options.get(job_name, None), node_count)
            gpu_count = self.parse_task_gpu_count(options.get("gpu", None), job_name, gpu_count_per_node)
            port_count = 1
            settings[job_name] = JobSetting(job_name, task_count, gpu_count, port_count)

        settings = [settings["ps"], settings["worker"]]
        self.adjust_job_setting_wild_card(settings, node_count, gpu_count_per_node)

        logging.info("Job Settings:")
        for job in settings:
            logging.info("\t%s: task_count=%s, port_count=%s, gpu_count=%s", job.name, job.task_count, job.port_count,
                         job.gpu_count)

        return settings

    def parse_job_task_count(self, input: str, node_count: int):
        if not input:
            return node_count

        if input == "*":
            return input

        value = float(input)
        if value == 0:
            return 0

        if value < 0:
            raise ValueError("invalid task count/percentage: %s" % input)

        if value.is_integer() and '.' not in input:
            return int(value)

        value = value * node_count
        if value.is_integer():
            return int(value)

        raise ValueError("invalid task percentage: %s" % str)

    def parse_task_gpu_count(self, input: str, job_name, gpu_count_per_node):
        if input == "*":
            return input

        if input:
            return int(input)

        if job_name == "ps":
            return 0
        else:
            return gpu_count_per_node

    def adjust_job_setting_wild_card(self, settings, node_count, gpu_count_per_node):
        ps_setting = settings[0]
        worker_setting = settings[1]

        if ps_setting.task_count == "*":
            ps_setting.task_count = node_count - worker_setting.task_count

        if worker_setting.task_count == "*":
            worker_setting.task_count = node_count - ps_setting.task_count

        if ps_setting.gpu_count == "*":
            ps_setting.gpu_count = gpu_count_per_node - worker_setting.gpu_count

        if worker_setting.gpu_count == "*":
            worker_setting.gpu_count = gpu_count_per_node - ps_setting.gpu_count

    def validate_job_settings(self, job_settings, node_count):
        total_task_count = sum(map(lambda s: int(s.task_count), job_settings))
        if node_count > total_task_count:
            raise RuntimeError("invalid total task count, node=%s, task=%s" % (node_count, total_task_count))

    def generate_task_settings(self, job_settings, node_list):
        def dequeue_n(l, n):
            if n == 0:
                return []

            if len(l) < n:
                raise RuntimeError("Not enought resource to allocate, requested:%s, left:%s" % (n, len(l)))

            tmp = l[:n]
            del l[:n]
            return tmp

        settings = []

        node_id = 0
        for job in job_settings:
            for task_index in range(job.task_count):
                node = node_list[node_id]
                logging.info("Allocate %s_%s on Node_%s", job.name, task_index, node.id)
                task_name = "%s_%s" % (job.name, task_index)
                port = dequeue_n(node.ports, job.port_count)[0]
                gpus = dequeue_n(node.gpus, job.gpu_count)
                settings.append(TaskSetting(node, job.name, task_name, task_index, port, gpus))
                node_id = (node_id + 1) % self.node_count

        return settings

    def print_task_settings(self, task_settings):
        def get_task_resource(task):
            return "%s: port=%s, gpus=%s" % (task.task_name, task.port, ','.join(task.gpus))

        node2tasks = {}
        for task in task_settings:
            node_id = task.node.id
            if node_id not in node2tasks:
                node2tasks[node_id] = [task]
            else:
                node2tasks[node_id].append(task)

        logging.info("Task settings:")
        for (node_id, tasks) in sorted(node2tasks.items(), key=lambda kv: (kv[0])):
            logging.info("\t[Node_%s] %s", node_id, '; '.join(map(get_task_resource, tasks)))

    def build_cluster_spec(self, job_settings, task_settings):
        spec = dict()
        for job in job_settings:
            spec[job.name] = []

        for task in task_settings:
            spec[task.job_name].append(task.node.ip + ':' + task.port)

        return spec

    def generate_mpi_launch_settings(self):
        node = self.current_node
        name = "worker_%s" % node.id
        argv = self.get_common_argv()
        cmd = util.get_normalized_argv(argv) + " " + self.apply_argument_variable(self.app_args_str, argv)
        envs = dict([self.get_cuda_visible_devices_env(node.gpus)])
        return [LaunchSetting(name, cmd, envs)]

    def generate_custom_launch_settings(self):
        node = self.current_node
        name = "worker_%s" % node.id
        argv = self.get_common_argv() + [
            self.format_arg_name(self.ARG_NODE_COUNT), str(self.node_count),
            self.format_arg_name(self.ARG_NODE_LIST_PATH), self.node_list_path,
            self.format_arg_name(self.ARG_NODE_ID), str(node.id),
            self.format_arg_name(self.ARG_NODE_NAME), node.name
        ]
        cmd = util.get_normalized_argv(argv) + " " + self.apply_argument_variable(self.app_args_str, argv)
        envs = dict([self.get_cuda_visible_devices_env(node.gpus)])
        return [LaunchSetting(name, cmd, envs)]

    def get_common_argv(self):
        argv = [self.python_path, "-u", self.app_script]
        argv += self.get_io_argv()
        if self.log_dir:
            argv += [self.format_arg_name(self.ARG_LOG_DIRECTORY), self.log_dir]
        return argv

    def get_io_argv(self):
        argv = []
        for arg in (self.input_args + self.output_args):
            (name, path) = arg
            if path is not None:
                argv += [self.format_arg_name(name), path]
        return argv

    def format_arg_name(self, name: str):
        if not self.use_underscore_arg:
            return name
        return util.convert_arg_dash_to_underscore(name)

    def apply_argument_variable(self, app_args: str, argv):
        if not app_args:
            return app_args

        arg_names = re.findall('\$\[(.+?)\]', app_args)
        if len(arg_names) > 0:
            arg_names = set(arg_names)
            used_arg_names = set(arg_names)

            parser = argparse.ArgumentParser()
            for name in arg_names:
                parser.add_argument("--" + name, nargs='?')
                name = util.convert_arg_dash_to_underscore(name)
                if name not in used_arg_names:
                    parser.add_argument("--" + name, nargs='?')
                    used_arg_names.add(name)

            (args, unknown) = parser.parse_known_args(argv)
            for name in arg_names:
                var = "$[%s]" % name
                value = util.get_arg_value(args, name)
                if value is None or value == "":
                    value = '""'
                app_args = app_args.replace(var, value)

        arg_names = re.findall('\[#(.+?)\]', app_args)
        if len(arg_names) > 0:
            arg_names = set(arg_names)
            used_arg_names = set(arg_names)

            parser = argparse.ArgumentParser()
            for name in arg_names:
                parser.add_argument("--" + name, nargs='?')
                name = util.convert_arg_dash_to_underscore(name)
                if name not in used_arg_names:
                    parser.add_argument("--" + name, nargs='?')
                    used_arg_names.add(name)

            (args, unknown) = parser.parse_known_args(argv)
            for name in arg_names:
                var = "[#%s]" % name
                value = util.get_arg_value(args, name)
                if value is None or value == "":
                    value = '""'
                app_args = app_args.replace(var, value)

        return app_args

    def get_cuda_visible_devices_env(self, gpus=None):
        if gpus is None or len(gpus) == 0:
            value = "-1"
        else:
            value = ','.join(gpus)

        return ("CUDA_VISIBLE_DEVICES", value)

    def launch(self):
        class AppRunContext(object):
            pass

        launch_settings = self.launch_settings
        app_count = len(launch_settings)
        if app_count == 0:
            raise RuntimeError("No app to launch")

        contexts = []
        for i in range(app_count):
            launch_setting = launch_settings[i]
            context = AppRunContext()
            context.id = i
            context.name = launch_setting.name
            context.cmd = launch_setting.cmd
            context.envs = launch_setting.envs
            contexts.append(context)

        for context in contexts:
            context.start_script = self.create_start_script(context.cmd, self.current_node.id, context.id)

        logging.info("Launching %s apps on %s:", app_count, self.current_node.full_name)

        cwd = self.app_dir
        # HACK: remove later
        if self.host_platform == consts.TRAINSTATION:
            cwd = os.getcwd()

        for context in contexts:
            env = dict(os.environ, **context.envs)

            # Add app directory to PYTHONPATH, it will be part of sys.path for launched script
            # So user can import module relative to app directory
            env["PYTHONPATH"] = self.app_dir

            if self.os_name == consts.OS_Windows:
                context.process = subprocess.Popen([context.start_script], shell=True, env=env, cwd=cwd)
            else:
                context.process = subprocess.Popen([context.start_script], shell=True, env=env, preexec_fn=os.setsid,
                                                   cwd=cwd)
            context.status = None
            logging.info("\t[%s] %s: envs=%s, start_script=%s, cmd=%s", context.process.pid, context.name, context.envs,
                         os.path.basename(context.start_script), context.cmd)

        succeed = 0
        failed = 0
        keep_waiting = True
        is_all_worker_succeed = False

        while keep_waiting:
            time.sleep(self.app_status_refresh_interval_in_seconds)
            for context in contexts:
                if context.status is not None:
                    continue
                status = context.process.poll()
                if status is not None:
                    context.status = status
                    if status == 0:
                        logging.info("%s succeed", context.name)
                        succeed += 1
                        if self.auto_kill_ps and context.name.startswith("worker_"):
                            self.save_worker_succeed_state(context.name)
                    else:
                        logging.error("%s failed, status=%s", context.name, status)
                        failed += 1

            if failed > 0:
                keep_waiting = False
            else:
                if not self.auto_kill_ps:
                    if succeed == app_count:
                        keep_waiting = False
                else:
                    if self.get_succeed_worker_count() == self.total_worker_count:
                        logging.info("All %s workers succeed", self.total_worker_count)
                        is_all_worker_succeed = True
                        keep_waiting = False

        succeed = 0
        failed = 0
        killed = 0
        logging.info("App final status on %s:", self.current_node.full_name)

        for context in contexts:
            status = context.process.poll()
            if status is None:
                try:
                    if self.os_name == consts.OS_Windows:
                        os.popen('TASKKILL /PID ' + str(context.process.pid) + ' /F')
                    else:
                        os.killpg(os.getpgid(context.process.pid), signal.SIGTERM)
                except OSError:
                    # Silently fail if the subprocess exits JUST before kill operation
                    pass
                logging.info("\t%s, killed", context.name)
                killed += 1
            elif status == 0:
                logging.info("\t%s, succeed", context.name)
                succeed += 1
            else:
                logging.info("\t%s, failed, status=%s", context.name, status)
                failed += 1

        logging.info("Succeed=%s, Failed=%s, Killed=%s", succeed, failed, killed)

        if self.sleep_after_finished_in_seconds > 0:
            logging.info("Sleep %s seconds", self.sleep_after_finished_in_seconds)
            time.sleep(self.sleep_after_finished_in_seconds)

        if not self.auto_kill_ps:
            return succeed == app_count
        else:
            return is_all_worker_succeed

    def create_start_script(self, cmd, node_id, launch_id):
        if self.os_name == consts.OS_Windows:
            path = os.path.join(self.runtime_dir, "start_%s_%s.bat" % (node_id, launch_id))
        else:
            path = os.path.join(self.runtime_dir, "start_%s_%s.sh" % (node_id, launch_id))

        if self.os_name == consts.OS_Windows:
            util.write_all_text(path, cmd)
        elif self.os_name == consts.OS_Linux:
            util.write_all_text(path, "#!/bin/sh" + os.linesep + cmd)
            os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)

        return path

    def save_worker_succeed_state(self, worker):
        pass

    def get_succeed_worker_count(self):
        pass

    def get_arg_value(self, arg, default=None):
        return util.get_arg_value(self.args, arg, default)
