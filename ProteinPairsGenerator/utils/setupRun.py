# General imports
import nvsmi
import os
from pathlib import Path
import time

def getFreeGPU(itrWait : float = 6e2, maxWait : float = 4.32e4):
    
    endTime = time.time() + maxWait

    while time.time() < endTime:

        # Get all free GPUs
        freeGPU = next(nvsmi.get_available_gpus(), None)

        # If there is an available GPU return this
        if not (freeGPU is None):
            return freeGPU

        # Else sleep and iterate further
        time.sleep(itrWait)

    return None

def setupRun(configFile : Path = Path("config.yaml"), runFile : Path = Path("../../../run.py"), **kwargs):

    # Get name
    name = "_".join(["{}={}".format(k, str(v)) for k, v in kwargs.items()])

    # Create folder structure
    root = Path("Experiments").joinpath(name)
    root.mkdir(parents=True)
    updatedConfig = root.joinpath("config.yaml")

    # Load config
    kwargs.update({"experiment_name": name})
    with updatedConfig.open("a") as out:
        for l in configFile.open().readlines():
            for k, v in kwargs.items():
                if k + ": " in l:
                    l = l.split(": ")[0] + ": " + str(v) + "\n"
                    break
            out.write(l)

    # Get free device
    device = getFreeGPU()

    # Run
    os.system("export CUDA_VISIBLE_DEVICES={} ; cd {} ; python3 {} --config config.yaml >> out.out".format(device.id, str(root), str(runFile)))
