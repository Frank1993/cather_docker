#!/usr/bin/python

import re
import os
import sys

from pyphillytools import phillylogger

if __name__ == "__main__":
    log_dir = str(os.path.dirname(os.path.realpath(__file__)))
    
    # Use only to test mixpanel
    #log_dir = '/var/storage/shared/phccp/sys/jobs/application_dummy/logs/'
    
    epoch_pattern = re.compile(r'^.*step\s490,\sloss\s=\s(?P<EPOCH_LOSS>.*?)\s.*$')
    minibatch_pattern = re.compile(r'^.*step\s\d+,\sloss\s=\s(?P<MINIBATCH_LOSS>.*?)\s.*$')
    command_pattern = re.compile(r'^.*Execute\sCommand:\s(?P<COMMAND>.*?)$')
    
    #pl = phillylogger(log_dir)
    #pl = phillylogger(log_dir, total_epoch=0)
    #pl = phillylogger(log_dir, total_epoch=0, total_minibatch=50)
    #pl = phillylogger(log_dir, 'command')
    #pl = phillylogger(log_dir, 'command', total_epoch=0)
    pl = phillylogger(log_dir, 'command', total_epoch=0, total_minibatch=50)
    
    if not pl.is_redirecting():
        exit(1)
    
    if 'application_dummy' not in log_dir:
        output_file = str(os.path.dirname(os.path.realpath(__file__))) + '/output'
        with open(output_file, 'r') as infile:
            for line in infile:
                print(line)
                match = epoch_pattern.match(line)
                if match:
                    pl.epoch_complete(float(match.group('EPOCH_LOSS')))
                match = minibatch_pattern.match(line)
                if match:
                    pl.minibatch_complete(float(match.group('MINIBATCH_LOSS')))
                match = command_pattern.match(line)
                if match:
                    #pl.new_command(match.group('COMMAND'))
                    #pl.new_command(match.group('COMMAND'), 1)
                    pl.new_command(match.group('COMMAND'), 1, 50)
                    pass
            pl.logging_complete()