import argparse
import os

from app_launcher.tensorflow_launcher import TensorFlowLauncher, NodeInfo
from shared import util


class TrainStationTensorFlowLauncher(TensorFlowLauncher):
    def init(self):
        super().init()
        self.is_offline_package_only = True
        self.auto_kill_ps = False

    def parse_node_list(self):
        node_list = []

        if self.is_local_mode:
            current_node_id = 0
            ports = ["10000", "10001"]
            node_list.append(NodeInfo(0, 0, util.get_host_name(), util.get_host_ip(), ports))
        else:
            parser = argparse.ArgumentParser()
            parser.add_argument(self.ARG_NODE_LIST_PATH, required=True)
            parser.add_argument(self.ARG_NODE_ID, required=True)

            (args, unknown) = parser.parse_known_args(self.argv)
            node_list_path = util.get_arg_value(args, self.ARG_NODE_LIST_PATH)
            current_node_id = int(util.get_arg_value(args, self.ARG_NODE_ID))

            if not os.path.isfile(node_list_path):
                raise FileNotFoundError("Node list file is missing, %s" % node_list_path)

            node_id = 0
            with open(node_list_path) as f:
                lines = f.read().splitlines()
                for line in [s.strip() for s in lines]:
                    if len(line) == 0:
                        continue

                    strs = line.split('\t')
                    name = strs[0]
                    ip = strs[1]
                    ports = strs[2].split(',')

                    node_list.append(NodeInfo(node_id, 0, name, ip, ports))
                    node_id += 1

        return node_list, current_node_id
