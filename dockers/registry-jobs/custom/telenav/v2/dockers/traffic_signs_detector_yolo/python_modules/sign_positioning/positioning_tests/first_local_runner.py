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
                                   ]
                                    ,cwd=self.app_path)
        process.wait()


    def __call__(self):
        print ("Run First Local App")
        self.run()
        print ("End First Local App")