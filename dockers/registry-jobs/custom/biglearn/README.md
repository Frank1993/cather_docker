
=================================
General Dockers for biglearn.
=================================

This docker is built for running biglearn training program

To training via biglearn, except for specify config and data dir. Users are also need to 
sepcify extra arguments 
    -r|--runtime <path> - BigLearn runtime package in zip, default /hdfs/<vc>/biglearn/runtimes/biglearn.zip
    -s|--sub-cmd <cmd> - the sub command to be executed on the given binary program, default train
    -e|--entry <binary> - the binary C# program to be executed, default BigLearn.CLI.exe
    --int-path <path> - the intermediate path to reference for intermediate data, default NONE
		* other arguments according to sub-cmd
