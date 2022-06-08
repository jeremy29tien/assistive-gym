#######################
# GPU UTILS           #
# (courtesy of Micah) #
#######################
import os
from logging import warning

import numpy as np


def get_freer_gpu():
    """
    This util is to try to figure out which GPU is most free, and automatically assign the current computation
    to that GPU. This probably only works with nvidia-smi GPUs.
    """
    os.system("nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    assert len(memory_available) > 0, "This is probably due to your "
    return np.argmax(memory_available)


def determine_default_torch_device(local):
    if local:
        _device_code = "cpu"
    else:
        try:
            GPU_ID = get_freer_gpu()
        except:
            warning(
                "Was not able to auto-assign the most free GPU to the job. Defaulting to the GPU 0"
            )
            GPU_ID = 0
        _device_code = "cuda:{}".format(GPU_ID)
    return _device_code
