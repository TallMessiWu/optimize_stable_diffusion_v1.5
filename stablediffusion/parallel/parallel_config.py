import os
from datetime import timedelta
import torch
import torch.distributed as dist
import torch_npu

class ParallelCfg:
    def __init__(self, enable_dp=False, device_id=0, local_rank=0, world_size=1):
        self.enable_dp = enable_dp
        self.device_id = device_id
        self.world_size = world_size
        self.rank = local_rank
        dist.init_process_group(
            backend="hccl", 
            rank=self.rank, 
            world_size=self.world_size,
            timeout=timedelta(minutes=30)
            )
        torch_npu.npu.set_device(self.device_id)
        