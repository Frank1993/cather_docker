import argparse

from app_launcher.tensorflow_launcher import LaunchSetting, NodeInfo, TensorFlowLauncher
from shared import util


class PAITensorFlowLauncher(TensorFlowLauncher):
    def init(self):
        self.argv += [
            "--input-data-dir", util.get_environment_variable("PAI_DATA_DIR"),
            "--output-dir", util.get_environment_variable("PAI_OUTPUT_DIR"),
        ]
        super().init()

        self.ip = util.get_environment_variable("PAI_CURRENT_CONTAINER_IP", self.ip)
        self.container_id = util.get_environment_variable("PAI_CONTAINER_ID")
        self.container_port = util.get_environment_variable("PAI_CURRENT_CONTAINER_PORT")
        self.total_role_count = int(util.get_environment_variable("PAI_TASK_ROLES_NUM"))
        self.role_id = int(util.get_environment_variable("PAI_TASK_ROLE_NO"))
        self.role_name = util.get_environment_variable("PAI_TASK_ROLE_NAME")
        self.total_task_count = int(util.get_environment_variable("PAI_TASKS_NUM"))
        self.task_count_in_role = int(util.get_environment_variable("PAI_TASK_ROLE_NUM"))
        self.task_id_in_role = int(util.get_environment_variable("PAI_TASK_ROLE_INDEX"))
        self.task_gpu_count = int(util.get_environment_variable("PAI_TASK_GPU_NUM"))
        self.role_host_list = []
        for i in range(self.total_role_count):
            self.role_host_list.append(util.get_environment_variable("PAI_TASK_ROLE_{}_HOST_LIST".format(i)))

        parser = argparse.ArgumentParser()
        parser.add_argument(self.ARG_EXE_MODE)
        (args, unknown) = parser.parse_known_args(self.argv)

        self.exe_mode = util.get_arg_value(args, self.ARG_EXE_MODE)
        if not self.exe_mode:
            if self.total_task_count == 1:
                self.exe_mode = self.EXE_MODE_SINGLETON
            elif self.total_role_count == 2:
                # TODO: Currently, we cannot tell whether it is a ps/worker mode, Platform support is required.
                # TODO: parse mpi mode when PAI is ready
                self.exe_mode = self.EXE_MODE_PS_WORKER
            else:
                raise RuntimeError

        self.auto_kill_ps = False
        self.use_underscore_arg = True

    def parse_node_list(self):
        node_list = []
        current_node_id = 0
        node_list.append(
            NodeInfo(0, 0, self.container_id, self.ip, [self.container_port], list(range(self.task_gpu_count))))
        return node_list, current_node_id

    def generate_ps_worker_launch_settings(self):
        is_worker = self.role_name.lower() == "worker"
        if is_worker:
            worker_role_id = self.role_id
        else:
            worker_role_id = 1 - self.role_id
        ps_role_id = 1 - worker_role_id

        cluster_spec = {"ps": self.role_host_list[ps_role_id], "worker": self.role_host_list[worker_role_id]}
        common_argv = self.get_common_argv()
        for (job_name, hosts) in cluster_spec.items():
            common_argv += [self.format_arg_name(self.ARG_TF_JOB_SPEC % job_name), hosts]

        job_name = "worker" if is_worker else "ps"
        task_name = "{}_{}".format(job_name, self.task_id_in_role)
        argv = common_argv + [
            self.format_arg_name(self.ARG_TF_JOB_NAME), job_name,
            self.format_arg_name(self.ARG_TF_TASK_INDEX), str(self.task_id_in_role)
        ]
        cmd = util.get_normalized_argv(argv) + " " + self.apply_argument_variable(self.app_args_str, argv)
        envs = dict([self.get_cuda_visible_devices_env(self.current_node.gpus)])
        return [LaunchSetting(task_name, cmd, envs)]
