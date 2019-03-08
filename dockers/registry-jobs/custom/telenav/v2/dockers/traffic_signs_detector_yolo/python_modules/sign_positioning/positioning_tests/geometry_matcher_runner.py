import subprocess

class GeometryMatcherRunner(object):

    def __init__(self, app_path, protobuf_input, map_path, output_file):
        self.app_path = app_path
        self.protobuf_input = protobuf_input
        self.output_file = output_file
        self.map_path = map_path

    def run(self):
        process = subprocess.Popen(["./geometry_matcher_app"
                                        ,"-i" + self.protobuf_input
                                        ,"-o" + self.output_file
                                        ,"-m" + self.map_path
                                   ]
                                    ,cwd=self.app_path)
        process.wait()

    def __call__(self):
        print "Run Geometry Matcher App"
        self.run()
        print "End Geometry Matcher App"