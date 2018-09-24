#!/usr/bin/python
import sys
import os
import re
import json

logdir=sys.argv[1]
cmd_name=sys.argv[2]
prog_path=os.path.join(logdir, 'progress.json')
log_path=os.path.join(logdir, 'logrank.0.log')
loss_pattern=re.compile(r'.*Training Loss (?P<L>[0-9\.]+), at iteration (?P<I>[0-9]+).*')
progress_pattern=re.compile(r'.*PROGRESS: (?P<P>[0-9\.]+)%.*')
max_loss=0
min_loss=0
epoch_losses=[]
loss=None
iter=None
prog=None
r=sys.stdin.readline()
while r:
  r=r.rstrip("\r\n")
  print(r)
  with open(log_path, 'a') as log:
    log.write(r+"\n")
  lp=loss_pattern.match(r)
  if lp:
    try:
      loss=float(lp.group('L'))
      iter=int(lp.group('I'))
      epoch_losses.append([iter, loss])
      if loss> max_loss:
        max_loss=loss
      elif loss<min_loss:
        min_loss=loss
    except:
      pass
  else:
    pg=progress_pattern.match(r)
    if pg:
      prog=float(pg.group('P'))
  if prog is not None:
    status={
        'lastErr':loss if loss else 0.00,
        'lastProgress':prog,
        'totEpochs':iter+1 if iter else 1,
        'gMinErr':min_loss,
        'gMaxErr':max_loss,
        'gFMinErr':min_loss,
        'gFMaxErr':max_loss,
        'logfilename':log_path,
        'curCommand':cmd_name,
        'commands':[{
          'name':cmd_name,
          'progress':prog,
          'totepoch':iter+1 if iter else 1,
          'minibatch':[],
          'finEpochs':epoch_losses
          }]
        }
    with open(prog_path, 'w') as fs:
      json.dump(status, fs, indent=2)
  prog=None
  try:
    r=sys.stdin.readline()
  except KeyboardInterrupt:
    break
