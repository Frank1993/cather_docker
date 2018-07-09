import horovod.torch as hvd
import torch
import sys

hvd.init()
rank = hvd.rank()
local = hvd.local_rank()
workers = hvd.size()
var = torch.LongTensor([0])
print("Worker %d of %d with local rank %d at barrier"%(rank+1,workers,local+1))
hvd.allreduce(var, name="Barrier")
print("Worker %d of %d past barrier"%(rank+1,workers))
# Exit with local rank as error code
sys.exit(local)