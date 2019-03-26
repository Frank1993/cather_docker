import argparse
import glob
import logging
import math
import os
import re
import time
from collections import namedtuple

from app_launcher.tensorflow_launcher import TensorFlowLauncher, NodeInfo
from shared import util

GPUInfo = namedtuple('GPUInfo', 'id name ecc')


class PhillyTensorFlowLauncher(TensorFlowLauncher):
    ContainerInfo = namedtuple('ContainerInfo', 'id name ip ssh_port')

    ARG_PHILLY_LOG_DIRECTORY = "--logDir"
    ARG_PORT_COUNT_PER_NODE = "--node-ports"
    ARG_NODE_PORT_BASE = "--node-port-base"
    ARG_NODE_PORT_BUCKET = "--node-port-bucket"
    ARG_USE_DIRECT_HDFS = "--direct-hdfs"
    ARG_COPY_PREV_MODEL_TO_OUTPUT = "--copy-prev-model-to-output"

    # Workaround:
    # A GPU on a Philly machine may show ecc error=2, tf always crash with segment fault, use --exclude-gpu-with-ecc-error to exclude such GPUs.
    # When using multiple Tesla K40 GPUs in a single process, tf may crash when model is large, use --single-tesla-k40 to use only 1 gpu for Tesla K40
    ARG_EXCLUDE_GPU_WITH_ECC_ERROR = "--exclude-gpu-with-ecc-error"
    ARG_USE_SINGLE_GPU_FOR_TESLA_K40 = "--single-tesla-k40"

    ARG_USE_RDMA = "--use-rdma"
    ARG_USE_NUMA = "--use-numa"

    def __init__(self, args, argv):
        super().__init__(args, argv)

    def init(self):
        super().init()

        parser = argparse.ArgumentParser()
        parser.add_argument(self.ARG_PHILLY_LOG_DIRECTORY)
        parser.add_argument(self.ARG_USE_DIRECT_HDFS, action="store_true", default=False)
        parser.add_argument(self.ARG_COPY_PREV_MODEL_TO_OUTPUT, action="store_true", default=False)
        parser.add_argument(self.ARG_EXCLUDE_GPU_WITH_ECC_ERROR, action="store_true", default=False)
        parser.add_argument(self.ARG_USE_SINGLE_GPU_FOR_TESLA_K40, action="store_true", default=False)
        parser.add_argument(self.ARG_USE_RDMA, action="store_true", default=False)
        parser.add_argument(self.ARG_USE_NUMA, nargs='?', const="all")
        (args, unknown) = parser.parse_known_args(self.argv)

        self.home_dir = os.path.expanduser("~")
        attempt_id = util.get_environment_variable("PHILLY_ATTEMPT_ID")
        self.state_dir = os.path.join(self.home_dir, "state", attempt_id)
        self.cluster_name = self.get_cluster_name()
        os.environ["PHILLY_CLUSTER"] = self.cluster_name
        self.use_direct_hdfs = util.get_arg_value(args, self.ARG_USE_DIRECT_HDFS)
        self.copy_prev_model_to_output = util.get_arg_value(args, self.ARG_COPY_PREV_MODEL_TO_OUTPUT)
        self.log_dir = util.get_arg_value(args, self.ARG_PHILLY_LOG_DIRECTORY)
        self.exclude_gpu_with_ecc_error = util.get_arg_value(args, self.ARG_EXCLUDE_GPU_WITH_ECC_ERROR)
        self.use_single_gpu_for_tesla_k40 = util.get_arg_value(args, self.ARG_USE_SINGLE_GPU_FOR_TESLA_K40)
        self.use_rdma = util.get_arg_value(args, self.ARG_USE_RDMA)
        self.use_numa = util.get_arg_value(args, self.ARG_USE_NUMA) if self.use_rdma else None
        self.gpu_info = self.load_gpu_info()

        self.print_io_paths()
        if self.use_direct_hdfs:
            self.convert_io_hdfs_paths()
            self.print_io_paths()

        if self.use_rdma:
            self.setup_rdma_envs()

    def get_cluster_name(self):
        if self.is_local_mode:
            return "local"

        _, output = util.execute_cmd("hostname -f")
        return output.split('.')[1]

    def load_gpu_info(self):
        if self.is_local_mode:
            return [GPUInfo(0, "FakeGPU", 0)]

        gpus = []
        output = os.popen('nvidia-smi --format=csv --query-gpu=index,name,ecc.errors.uncorrected.volatile.total').read()
        logging.info("[%s] nvidia-smi --format=csv --query-gpu=index,name,ecc.errors.uncorrected.volatile.total: \n%s",
                     self.ip, output)

        for line in output.splitlines()[1:]:
            line = line.strip()
            atributes = line.split(', ')
            id = int(atributes[0])
            name = atributes[1]
            ecc = util.try_parse_int(atributes[2], 0)
            gpus.append(GPUInfo(id, name, ecc))

        return gpus

    def convert_io_hdfs_paths(self):
        logging.info("Convert local path to hdfs path:")
        self.input_args = [(name, self.convert_hdfs_path(path)) for (name, path) in self.input_args]
        self.output_args = [(name, self.convert_hdfs_path(path)) for (name, path) in self.output_args]
        self.log_dir = self.convert_hdfs_path(self.log_dir)

    def convert_hdfs_path(self, path: str):
        if not path or not path.startswith("/hdfs/"):
            return path
        return path.replace('/hdfs/', 'hdfs://%s/' % self.cluster_name, 1)

    def parse_node_list(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(self.ARG_PORT_COUNT_PER_NODE, nargs='?')
        parser.add_argument(self.ARG_NODE_PORT_BASE)
        parser.add_argument(self.ARG_NODE_PORT_BUCKET)

        (args, unknown) = parser.parse_known_args(self.argv)
        port_count_per_node = int(util.get_arg_value(args, self.ARG_PORT_COUNT_PER_NODE, 2))
        port_base = int(util.get_arg_value(args, self.ARG_NODE_PORT_BASE, 10000))
        port_bucket = int(util.get_arg_value(args, self.ARG_NODE_PORT_BUCKET, 128))

        logging.info("Construct node list")
        logging.info("\tport_count_per_node=%s", port_count_per_node)
        logging.info("\tport_base=%s", port_base)
        logging.info("\tport_bucket=%s", port_bucket)

        node_list = []
        if self.is_local_mode:
            current_node_id = 0
            node = NodeInfo(0, 0, util.get_host_name(), self.ip,
                            self.generate_ports(port_base, port_count_per_node), [])
            node_list.append(node)
        else:
            world_size = util.get_mpi_word_size()
            local_size = util.get_mpi_local_size()
            local_rank = util.get_mpi_local_rank()

            container_count = int(world_size / local_size)
            container_list = self.parse_container_list()
            current_container = self.get_current_container(container_list)

            if container_count != len(container_list):
                raise RuntimeError(
                    "Parsed container list length is %s, but %s expected" % (len(container_list), container_count))

            gpu_count_per_container = len(self.gpu_info)
            logging.info("Found %s containers, each has %s GPUs", container_count, gpu_count_per_container)
            for container in container_list:
                logging.info("\t[%s] %s, ip=%s, ssh_port=%s", container.id, container.name, container.ip,
                             container.ssh_port)

            node_count_per_container = local_size
            gpu_count_per_node = int(gpu_count_per_container / node_count_per_container)

            node_id = 0
            current_node_id = -1
            for container in container_list:
                container_port_base = ((container.ssh_port * port_bucket) % port_base) + port_base
                is_current_container = (container.id == current_container.id)
                for i in range(node_count_per_container):
                    name = container.name + "_" + str(i)
                    ports = self.generate_ports(container_port_base + i * port_count_per_node, port_count_per_node)
                    gpus = list(range(i * gpu_count_per_node, (i + 1) * gpu_count_per_node))
                    node = NodeInfo(node_id, i, name, container.ip, ports, gpus)
                    node_list.append(node)
                    if is_current_container and i == local_rank:
                        current_node_id = node.id
                    node_id += 1

            current_node = node_list[current_node_id]
            if current_node.is_local_master():
                util.create_directory(self.state_dir)

            if self.use_rdma:
                self.update_node_list_ib_ip(node_list, container_count, current_node)

        return node_list, current_node_id

    def parse_container_list(self):
        container_ips = []
        etc_hosts_path = os.path.join(self.home_dir, "etc-hosts")
        for line in util.read_all_text(etc_hosts_path).splitlines():
            strs = line.split(' ')
            if len(strs) > 1:
                name = strs[1]
                ip = strs[0]
                container_ips.append((name, ip))
        container_ips.sort(key=lambda item: item[0])

        container_ports = []
        ssh_config_path = os.path.join(self.home_dir, ".ssh", "config")
        current_container = ""
        for line in util.read_all_text(ssh_config_path).splitlines():
            line = line.strip()
            if line.startswith("Host"):
                current_container = line.split(' ')[1]
            elif line.startswith("Port"):
                container_ports.append((current_container, line.split(' ')[1]))
        container_ports.sort(key=lambda item: item[0])

        if len(container_ips) != len(container_ports):
            raise RuntimeError("Container count mismatch, ~/etc-hosts=%s, ~/.ssh/config=%s" %
                               (len(container_ips), len(container_ports)))

        container_list = []
        for i in range(len(container_ips)):
            if container_ips[i][0] != container_ports[i][0]:
                raise RuntimeError("Container name mismatch, %s, %s" % (container_ips[i][0], container_ports[i][0]))

            name = container_ports[i][0]
            ip = container_ips[i][1]
            ssh_port = int(container_ports[i][1])
            container_list.append(self.ContainerInfo(i, name, ip, ssh_port))

        return container_list

    def get_current_container(self, container_list):
        ssh_client = util.get_environment_variable("SSH_CLIENT")
        ssh_port = int(ssh_client.strip().split(' ')[-1])

        for container in container_list:
            if self.ip == container.ip and ssh_port == container.ssh_port:
                return container

        raise RuntimeError("Failed to find current container name for %s:%s" % (self.ip, ssh_port))

    def generate_ports(self, base: int, count: int):
        ports = []
        for i in range(count):
            ports.append(base + i)
        return ports

    def get_offline_pip_package_dirs(self):
        root = "/hdfs/public/apprunner/packages/pip"
        return [os.path.join(root, "pypi"), os.path.join(root, "custom")]

    def generate_ps_worker_launch_settings(self):
        settings = super().generate_ps_worker_launch_settings()
        if not self.use_rdma:
            return settings

        logging.info("Update launch setting for NUMA")
        output = os.popen('numactl --hardware').read()
        logging.info("numactl --hardware: \n%s", output)

        m = re.search('available: (.+?) nodes', output)
        numa_node_count = int(m.group(1))

        cpu_count = 0
        node_cpus = []
        for i in range(numa_node_count):
            m = re.search('node %d cpus: (.+)' % i, output)
            cpus = m.group(1).split(' ')
            node_cpus.append(cpus)
            cpu_count += len(cpus)

        node_masks = []
        for i in range(numa_node_count):
            mask = ['0'] * cpu_count
            for cpu in node_cpus[i]:
                mask[int(cpu)] = '1'
            mask.reverse()
            node_masks.append(str(hex(int("".join(mask), 2))))

        if self.use_numa == "ib":
            logging.info("Get IB NUMA node")
            output = os.popen('nvidia-smi topo -m').read()
            logging.info("nvidia-smi topo -m: \n%s", output)

            lines = output.splitlines()

            header = [x.strip() for x in lines[0].split('\t')]
            device = os.environ["RDMA_DEVICE"]
            ib_id = header.index(device)

            for line in lines[1:]:
                strs = [x.strip() for x in line.split('\t')]
                print(strs)
                if strs[ib_id] == "PHB" or strs[ib_id] == "PIX":
                    cpu_range = strs[-1].split('-')
                    break

            for i in range(numa_node_count):
                cpus = node_cpus[i]
                if cpus[0] == cpu_range[0] and cpus[-1] == cpu_range[-1]:
                    numa_node_id = i

            logging.info("Use NUMA node %s", numa_node_id)
            for setting in settings:
                setting.cmd = "taskset %s %s" % (node_masks[numa_node_id], setting.cmd)
        else:
            numa_node_id = self.current_node.id % numa_node_count
            for setting in settings:
                setting.cmd = "taskset %s %s" % (node_masks[numa_node_id], setting.cmd)
                numa_node_id = (numa_node_id + 1) % numa_node_count

        return settings

    def get_cuda_visible_devices_env(self, gpus=None):
        if gpus is not None and len(gpus) > 0:
            if self.exclude_gpu_with_ecc_error:
                logging.info("Exclude gpus with ecc error")
                valid_gpus = []
                for gpu_id in gpus:
                    ecc = self.gpu_info[int(gpu_id)].ecc
                    if ecc == 0:
                        valid_gpus.append(gpu_id)
                    else:
                        logging.warning("Exclude gpu_%s with ecc=%s", gpu_id, ecc)

                if len(valid_gpus) == 0:
                    raise RuntimeError("All gpus are excluded")
                gpus = valid_gpus

            if self.use_single_gpu_for_tesla_k40 and "Tesla K40" in self.gpu_info[0].name:
                logging.warning("Use only 1 GPU for Tesla K40")
                gpus = gpus[0:1]

        return super().get_cuda_visible_devices_env(gpus)

    def setup(self):
        super().setup()

        # This is a workaround for backward compatibility
        done_path = os.path.join(self.home_dir, "prev_model_copy.done")
        if self.copy_prev_model_to_output and not os.path.isfile(done_path):
            prev_model_src = next((x[1] for x in self.input_args if x[0] == "--input-previous-model-path"), None)
            prev_model_dest = next((x[1] for x in self.output_args if x[0] == "--output-model-path"), None)
            if prev_model_src and prev_model_dest:
                if self.current_node.is_master():
                    if not self.use_direct_hdfs:
                        prev_model_src = self.convert_hdfs_path(prev_model_src)
                        prev_model_dest = self.convert_hdfs_path(prev_model_dest)

                    logging.info("Copy prev model on master node")
                    util.execute_cmd("hdfs dfs -cp %s %s" % (prev_model_src, prev_model_dest))
                    util.write_all_text(done_path)
                else:
                    logging.info("Wait master to copy prev model")
                    if not util.wait_file(done_path):
                        raise RuntimeError("Wait master to copy prev model timeout")

    def save_worker_succeed_state(self, worker):
        state_path = os.path.join(self.state_dir, "%s.succeed" % worker)
        util.write_all_text(state_path)

    def get_succeed_worker_count(self):
        path = os.path.join(self.state_dir, "worker_*.succeed")
        return len(glob.glob(path))

    # This is a workaround for wolong rdma to decide which rdma device to use
    def setup_rdma_envs(self):
        output = os.popen('ibstat').read()
        if "mlx4_" in output:
            device = "mlx4_0"
            port = "2"
        elif "mlx5_" in output:
            device = "mlx5_0"
            port = "1"
        else:
            logging.warning("Failed to identify rdma device, ibstat=%s", output)
            return

        os.environ["RDMA_DEVICE"] = device
        os.environ["RDMA_PORT"] = port
        logging.info("Set RDMA_DEVICE=%s, RDMA_PORT=%s", device, port)

    def update_node_list_ib_ip(self, node_list, container_count, current_node):
        logging.info("Update node ip for rdma")
        self.print_node_list(node_list)

        if current_node.is_local_master():
            state_path = os.path.join(self.state_dir, "ib_%s" % current_node.name)
            util.write_all_text(state_path, self.ip + '\t' + util.get_host_ib_ip())

        logging.info("Wait ib states")
        files = self.wait_files(self.state_dir, "ib_*", container_count)
        if len(files) != container_count:
            raise RuntimeError("Wait ib state failed, expected=%d, found=%d" % (container_count, len(files)))

        ip_ibs = dict()
        for file in files:
            strs = util.read_all_text(file).splitlines()[0].split('\t')
            ip = strs[0]
            ip_ib = strs[1]
            ip_ibs[ip] = ip_ib

        for node in node_list:
            node.ip = ip_ibs[node.ip]

    def wait_files(self, dir, pattern, count: int = 1, interval: int = 1, timeout: int = 300):
        path_name = os.path.join(dir, pattern)
        loop = math.ceil(timeout / interval)
        while loop > 0:
            files = glob.glob(path_name)
            if len(files) == count:
                return files
            time.sleep(interval)
            loop -= 1
        return []
