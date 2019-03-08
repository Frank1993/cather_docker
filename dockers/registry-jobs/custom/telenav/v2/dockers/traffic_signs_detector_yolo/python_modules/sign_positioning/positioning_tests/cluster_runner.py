import subprocess

class ClusterRunner(object):
    def __init__(self, app_path, input_roi_file, input_cluster_file, output_file):
        self.app_path = app_path
        self.input_roi_file = input_roi_file
        self.input_cluster_file = input_cluster_file
        self.output_file = output_file

    def run(self):
        if self.input_cluster_file is '':
            process = subprocess.Popen(["./cluster_app", "-i" + self.input_roi_file,
            "-o" + self.output_file], cwd=self.app_path)
        else:    
            process = subprocess.Popen(["./cluster_app", "-i" + self.input_roi_file
            , "-c" + self.input_cluster_file, "-o" + self.output_file], cwd=self.app_path)
        process.wait()

    def __call__(self):
        print ("Run Cluster App")
        self.run()
        print ("End Cluster App")