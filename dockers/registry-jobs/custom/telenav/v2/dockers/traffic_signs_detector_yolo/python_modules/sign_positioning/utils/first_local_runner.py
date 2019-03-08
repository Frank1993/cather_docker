import subprocess


class FirstLocalRunner(object):

    def __init__(self, app_path, input_file, output_file):
        self.app_path = app_path
        self.input_file = input_file
        self.output_file = output_file

    def run(self):
        process = subprocess.Popen(["./first_local_app"
                                        ,"-i" + self.input_file
                                        ,"-o" + self.output_file
                                   ],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.app_path)
        error_data = process.communicate()
        return_code = process.returncode
        return return_code, error_data

    def __call__(self):
        return self.run()