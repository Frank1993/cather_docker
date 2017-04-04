#!/usr/bin/python
from __future__ import print_function

import re
import os
import sys

valid_triggers = ['COMMAND', 'PROGRESS', 'EPOCH_LOSS', 'MINIBATCH_LOSS']
valid_variables = ['COMMAND', 'TOTAL_EPOCH', 'TOTAL_MINIBATCH', 'CURRENT_EPOCH', 'CURRENT_MINIBATCH', 'EPOCH_LOSS', 'MINIBATCH_LOSS']

class phillylogger:
    def __init__(self, filename, total_epoch=None, total_minibatch=None):       
        self.variables = {}
        self.parser_dict = None
        self.trigger_dict = None
        self.filename = filename  
        self._stdout = sys.stdout
        
        # Setup the default values
        self.variables['TOTAL_EPOCH'] = 0
        self.variables['TOTAL_MINIBATCH'] = 0
        self.variables['CURRENT_EPOCH'] = 0
        self.variables['CURRENT_MINIBATCH'] = 0
        self.variables['EPOCH_LOSS'] = 0.0
        self.variables['MINIBATCH_LOSS'] = 0.0
        
        if not os.path.isdir(os.path.dirname(self.filename)):
            print("ERROR: Invalid directory (" + directory + ")")
            exit(1)
        if os.path.exists(self.filename):
            os.remove(self.filename)
        
        if total_epoch:
            self.update_total_epoch(total_epoch)
        if total_minibatch:
            self.update_total_epoch(total_minibatch)

    def run(self):        
        if not self.parser_dict:
            print("ERROR: Parser dictionary not setup.")
            exit (1)  
        if not self.trigger_dict:
            print("ERROR: Trigger dictionary not setup.")
            exit (1)
                
        print("Redirecting stdout to " + self.filename)
        sys.stdout = self
    
    def write(self, buf):
        with open(self.filename, "a") as myfile:    
            for line in buf.rstrip().splitlines():
                myfile.write(line + "\n")
                
                # Parse the variables
                for variable in valid_variables:
                    if variable in self.parser_dict:
                        pattern = re.compile(self.parser_dict[variable])
                        match = pattern.match(line)
                        if match:
                            self.variables[variable] = match.group(variable)
                            #myfile.write("UPDATE: " + variable + ": " + self.variables[variable] + "\n")
                
                # Process the print triggers
                for trigger in valid_triggers:
                    if trigger in self.trigger_dict:
                        pattern = re.compile(self.trigger_dict[trigger])
                        match = pattern.match(line)
                        if match:
                            if trigger == 'PROGRESS':
                                percent = 0
                                if self.variables['TOTAL_EPOCH'] > 0:
                                    percent += float(self.variables['CURRENT_EPOCH']) / float(self.variables['TOTAL_EPOCH'])                                
                                if self.variables['TOTAL_MINIBATCH'] > 0 and self.variables['TOTAL_EPOCH'] > 0:
                                    mini_percent = float(self.variables['CURRENT_MINIBATCH']) / float(self.variables['TOTAL_MINIBATCH'])
                                    percent += (1.0 / float(self.variables['TOTAL_EPOCH'])) * mini_percent
                                percent *= 100
                                if percent > 100:
                                    percent = 100
                                myfile.write("PROGRESS: %.2f%%\n" % percent)
                                print("PROGRESS: %.2f%%" % percent, file=self._stdout)
                            elif trigger == 'EPOCH_LOSS':
                                loss = float(self.variables['EPOCH_LOSS'])
                                myfile.write("EPOCH_LOSS: %.7f%%\n" % loss)
                                print("EVALERR: %.7f%%" % loss, file=self._stdout)
                            elif trigger == 'MINIBATCH_LOSS':
                                loss = float(self.variables['MINIBATCH_LOSS'])
                                myfile.write("MINIBATCH_LOSS: %.7f%%\n" % loss)
                                print("EVALERR: %.7f%%" % loss, file=self._stdout)
                            else:
                                myfile.write(trigger + ": " + self.variables[trigger] + "\n")
    
    def setup_parser(self, parser_dict):
        if type(parser_dict) is dict:
            if self.parser_dict:
                print("ERROR: Parsing dictionary already setup")
                exit(1)
            self.parser_dict = parser_dict
            for variable in valid_variables:
                if variable not in self.parser_dict:
                    print("WARNING: Missing the variable " + variable + ". Web portal display may not work correctly")
        else:
            print("ERROR: setup_parser() expects a dictionary type")
            exit(1)
            
    def setup_trigger(self, trigger_dict):
        if type(trigger_dict) is dict:
            if self.trigger_dict:
                print("ERROR: Trigger dictionary already setup")
                exit(1)
            self.trigger_dict = trigger_dict
            for trigger in valid_triggers:
                if trigger not in self.trigger_dict:
                    print("WARNING: Missing the trigger " + trigger + ". Web portal display may not work correctly")
        else:
            print("ERROR: setup_trigger() expects a dictionary type")
            exit(1)
    
    def update_total_epoch(self, total_epoch):
        self.variables['TOTAL_EPOCH'] = total_epoch
        with open(self.filename, "a") as myfile:
            myfile.write("TOTAL_EPOCH: " + str(self.variables['TOTAL_EPOCH']) + "\n")
    
    def update_total_minibatch(self, total_minibatch):
        self.variables['TOTAL_MINIBATCH'] = total_minibatch
        with open(self.filename, "a") as myfile:
            myfile.write("TOTAL_MINIBATCH: " + str(self.variables['TOTAL_MINIBATCH']) + "\n")