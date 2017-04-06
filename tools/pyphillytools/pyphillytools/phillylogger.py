#!/usr/bin/python
from __future__ import print_function

import re
import os
import sys
import json
import _ctypes

from pyphillytools import FileLock

try:
    from mixpanel import Mixpanel
except:
    print('WARNING: Either mixpanel or json not installed')

valid_triggers = ['EPOCH_COMPLETE', 'MINIBATCH_COMPLETE', 'COMMAND']

# Mixpanel is used to record events.  For here it will record
# the event everytime that this module is initialized
mixpanel_token = '4820569f5a33e95d481ef207a4e769d8'
def mixpanel(filename):
    pattern = re.compile(r'^/var/storage/shared/(?P<VC>.*?)/sys/jobs/(?P<APP>.*?)/logs.*$')
    match = pattern.match(filename)
    
    if not match.group('VC') or not match.group('APP'):
        print('WARNING: Invalid application directory (' + filename + ')')
        return
    
    json_file = '/var/storage/shared/' + match.group('VC') + '/sys/jobs/' + match.group('APP') + '/metadata.json'
    
    with open(json_file) as data_file:
        data = json.load(data_file)
        user = data['user']
    
        # add a build event to pixpanel
        mp = Mixpanel(mixpanel_token)
        mp.track(user, 'Started Logging Tool', data)
        mp.people_set(user, {'$email': user + '@microsoft.com'})

class phillylogger:
    def __init__(self, log_directory, command=None, total_epoch=None, total_minibatch=None):
        self.trigger_patterns = {}
        self._stdout = sys.stdout
        self.redirecting = False
        
        # Setup the default values
        self.command = 'no-command'
        self.progress = 0.0
        self.total_epoch = None
        self.total_minibatch = None
        self.epoch_loss = 0.0
        self.minibatch_loss = 0.0
        self.current_epoch = 0.0
        self.current_minibatch = 0.0
        self.last_err = 0.0
        self.min_epoch = 10000.0
        self.max_epoch = 0.0
        self.min_minibatch = 10000.0
        self.max_minibatch = 0.0
                
        # Setup auto increment for totals
        self.auto_epoch = False
        if not self.total_epoch:
            self.auto_epoch = True
            self.total_epoch = 1.0
        
        # Print Warning
        if not total_epoch and total_minibatch:
            print("WARNING: Total Minibatch given without Total Epoch, ignoring Minibatch.", file=self._stdout)
            total_minibatch = None

        # Test that the directory is valid
        if not os.path.isdir(log_directory):
            print("ERROR: Invalid directory (" + log_directory + ")")
            print("       The logging module cannot continue")
            return None
        
        # Create the two filenames
        self.log_filename = os.path.join(log_directory, "redirected_stdout.log")
        self.json_filename = os.path.join(log_directory, "progress.json")
        
        # Setup locks for file writing
        log_lock = FileLock(self.log_filename)
        json_lock = FileLock(self.json_filename)
        
        # Setup blank json obj
        self.commands = []
        self.commandIndex = -1
        if command: self.command = command
        self.first_run = True
        self.new_command(self.command, total_epoch, total_minibatch)
        
        # Remove the files if they exist, should never happen
        if os.path.exists(self.log_filename):
            os.remove(self.log_filename)
        if os.path.exists(self.json_filename):
            os.remove(self.json_filename)

        # Try to send mixpanel event
        try:
            mixpanel(self.filename)
        except:
            pass

        # Start the stdout redirection
        print("Redirecting output to " + self.log_filename)
        print("Saving progress data to " + self.json_filename)
        self.redirecting = True
        sys.stdout = self
    
    # Cleanup
    def __exit__(self):
        sys.stdout = self._stdout
    
    # Query if this module is redirecting
    def is_redirecting(self):
        return self.redirecting
    
    # Save the json file (overwrites if already exists)
    def save_json(self):
        temp_json = {'curCommand': self.command,
                     'lastErr': self.last_err,
                     'gFMinErr': self.min_epoch,
                     'gFMaxErr': self.max_epoch,
                     'logfilename': self.log_filename,
                     'gMMinErr': self.min_minibatch,
                     'gMMaxErr': self.max_minibatch,
                     'lastProgress': self.progress,
                     'totEpochs': self.total_epoch,
                     'commands': self.commands}
        with open(self.json_filename, 'w') as jsonFile:
            with FileLock(self.json_filename) as fl:
                json.dump(temp_json, jsonFile) #, indent=4) Add for pretty printing
    
    # Process the new command information
    def new_command(self, command, total_epoch=None, total_minibatch=None):
        if not total_epoch and total_minibatch:
            print("WARNING: Total Minibatch given without Total Epoch, ignoring Minibatch.")
            total_minibatch = None
        
        if self.first_run:
            if self.total_epoch and not total_epoch:
                print("WARNING: No Total Epoch supplied dispite constructor arguments, unpredictable results")
            if self.total_minibatch and not total_minibatch:
                print("WARNING: No Total Minibatch supplied dispite constructor arguments, unpredictable results")
            if self.total_epoch and not self.total_minibatch and total_minibatch:
                print("WARNING: No Total Minibatch supplied dispite constructor arguments, unpredictable results")
            self.first_run = False
        
        # Auto epoch has an inherit +1 of total epochs
        # Remove and turn off if received a total_epoch
        if total_epoch and self.auto_epoch:
            print("WARNING: Auto increment epoch was on, unpredictable results")
            self.total_epoch -= 1.0
            self.auto_epoch = False
        
        # Add to the total epochs
        if total_epoch:
            self.total_epoch += float(total_epoch)
        
        # Replace the current minibatch
        if total_minibatch:
            self.total_minibatch = float(total_minibatch)
        else:
            # Might get array index error without
            self.total_minibatch = None
        
        
        if self.commandIndex >= 0:
            eLen = len(self.commands[self.commandIndex]['finEpochs'])
            bLen = len(self.commands[self.commandIndex]['minibatch'])
            self.commands[self.commandIndex]['totepoch'] = float(eLen)
            # Romove the command if there is no data
            if eLen + bLen <= 0:
                del self.commands[self.commandIndex]
                self.commandIndex -= 1
            else:
                # The following will pad the values in the graph for completeness
                if eLen > 0:
                    if self.commandIndex == 0 and len(self.commands[self.commandIndex]['finEpochs']) >= 0:
                        firstEpoch = self.commands[self.commandIndex]['finEpochs'][0][1]
                        self.commands[self.commandIndex]['finEpochs'].insert(0, [0.0, firstEpoch])
                    elif self.commandIndex > 0 and len(self.commands[self.commandIndex-1]['finEpochs']) >= 0:
                        lastEpochPair = self.commands[self.commandIndex-1]['finEpochs'][-1]
                        self.commands[self.commandIndex]['finEpochs'].insert(0, lastEpochPair)
                if bLen > 0:
                    if self.commandIndex == 0 and len(self.commands[self.commandIndex]['minibatch']) >= 0:
                        firstMinibatch = self.commands[self.commandIndex]['minibatch'][0][1]
                        self.commands[self.commandIndex]['minibatch'].insert(0, [0.0, firstMinibatch])
                    elif self.commandIndex > 0 and len(self.commands[self.commandIndex-1]['minibatch']) >= 0:
                        lastMinibatchPair = self.commands[self.commandIndex-1]['minibatch'][-1]
                        self.commands[self.commandIndex]['minibatch'].insert(0, lastMinibatchPair)
        
        # Update and ssve the json
        self.update_command()
        self.update_progress()
        self.save_json()
        
        # Create a new command
        self.command = command
        self.commandIndex += 1
        self.commands.append({'progress': self.progress,
                              'totepoch': self.total_epoch,
                              'name': self.command,
                              'finEpochs': [],
                              'minibatch': []})           
    
    # Epoch complete trigger
    def epoch_complete(self, loss):
        self.current_epoch += 1.0
        self.current_minibatch = 0.0
        self.epoch_loss = float(loss)
        self.progress = self.calculate_progress()
        print("PROGRESS: %.2f%%" % self.progress, file=self._stdout)
        print("EVALERR: %.7f%%" % self.epoch_loss, file=self._stdout)
            
        # Update min & max
        if self.min_epoch > self.epoch_loss:
            self.min_epoch = self.epoch_loss
        if self.max_epoch < self.epoch_loss:
            self.max_epoch = self.epoch_loss
        self.last_err = self.epoch_loss
        
        # Update the finEpochs
        temp_value = round(self.progress * self.total_epoch / 100.0, 1)
        self.commands[self.commandIndex]['finEpochs'].append([temp_value,self.epoch_loss])
        
        # Save the json file
        self.save_json()
        
        # Auto increment
        if self.auto_epoch and self.total_epoch < self.current_epoch - 1:
            self.total_epoch = self.current_epoch
    
    # Minibatch complete trigger
    def minibatch_complete(self, loss):
        if self.total_epoch:
            self.current_minibatch += 1.0
            self.minibatch_loss = float(loss)
            self.progress = self.calculate_progress()
            print("PROGRESS: %.2f%%" % self.progress, file=self._stdout)
            print("EVALERR: %.7f%%" % self.minibatch_loss, file=self._stdout)
            
            if self.total_minibatch: 
                # Update min & max
                if self.min_minibatch > self.minibatch_loss:
                    self.min_minibatch = self.minibatch_loss
                if self.max_minibatch < self.minibatch_loss:
                    self.max_minibatch = self.minibatch_loss
                self.last_err = self.minibatch_loss
                
                # Update the minibatch
                temp_value = round(self.progress * self.total_epoch / 100.0, 6)
                self.commands[self.commandIndex]['minibatch'].append([temp_value,self.minibatch_loss])
            
                # Save the json file
                self.save_json()
            
    # Update the current command only if auto_epoch
    def update_command(self):
        if self.auto_epoch and self.commandIndex >= 0:
            command = self.commands[self.commandIndex]
            if len(command['finEpochs']) > 0:
                start_epoch = command['finEpochs'][0][0]
                current_epoch = start_epoch
                # Recalculate the epoch numbers
                for epoch in command['finEpochs']:
                    epoch[0] = current_epoch
                    current_epoch += 1
                if len(command['minibatch']) > 0:
                    # Recalculate the minibatch numbers
                    current_epoch = start_epoch
                    mini_count = float(len(command['minibatch'])) - 1.0
                    for minibatch in command['minibatch']:
                        minibatch[0] = round(current_epoch, 2)
                        current_epoch += command['totepoch'] / mini_count
            else:
                print("WARNING: Missing elements in finEpochs, ignoring", file=self._stdout)
            
    # Update progress for each command
    def update_progress(self):
        progress = 0.0
        for command in self.commands:
            command['progress'] = round(progress * 100, 2)
            progress += float(command['totepoch'] / self.total_epoch)
    
    # Command complete trigger
    def logging_complete(self):
        self.new_command('dummy')
    
    # Calculate the progress
    def calculate_progress(self):
        percent = 0.0
        if self.total_epoch > 0:
            percent += self.current_epoch / self.total_epoch  
        if self.total_epoch > 0 and self.total_minibatch > 0:
            mini_percent = self.current_minibatch / self.total_minibatch
            percent += mini_percent / self.total_epoch
        percent *= 100.0
        if percent > 100.0:
            percent = 100.0
        return round(percent, 2)
    
    # Process the stdout lines
    def write(self, buf):
        with open(self.log_filename, "a") as myfile:
            with FileLock(self.log_filename) as fl:
                for line in buf.rstrip().splitlines():
                    myfile.write(line + '\n')
