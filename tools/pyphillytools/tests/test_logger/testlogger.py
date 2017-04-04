#!/usr/bin/python

import re
import os

from pyphillytools import phillylogger

valid_triggers = ['PROGRESS', 'EPOCH_LOSS', 'MINIBATCH_LOSS', 'EVALERR']
valid_variables = ['TOTAL_EPOCH', 'TOTAL_MINIBATCH', 'EPOCH', 'MINIBATCH', 'EPOCH_LOSS', 'MINIBATCH_LOSS']

if __name__ == "__main__":
    print("This should go to stdout")
    
    pl = phillylogger(str(os.getcwd()) + '/log.log', 110)
    pl.setup_parser ({'CURRENT_EPOCH': r'^.*Start\sEpoch\s(?P<CURRENT_EPOCH>.*?)\s.*$',
                      'EPOCH_LOSS': r'^.*l2_regularize_loss:\s(?P<EPOCH_LOSS>.*?)$',
                      'TOTAL_MINIBATCH': r'^.*\|\d+\/(?P<TOTAL_MINIBATCH>.*?)\[.*$',
                      'CURRENT_MINIBATCH': r'^.*\|(?P<CURRENT_MINIBATCH>.*?)\/\d+\[.*$',
                      'COMMAND': r'^.*Start\sEpoch\s(?P<COMMAND>.*?)\s.*$'})
    pl.setup_trigger({'PROGRESS': r'^.*Epoch\s\d+\s\(.*$',
                      'PROGRESS': r'^.*\|\d+\/\d+\[.*$',
                      'EPOCH_LOSS': r'^.*l2_regularize_loss.*$'})
    pl.run()
    
    with open('output', 'r') as infile:
        for line in infile:
            print(line)