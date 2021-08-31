# General imports
from mlflow.tracking import MlflowClient
from multiprocessing import Lock
import nvsmi
import os
from pathlib import Path
from random import randint
import time
from typing import Union

locks = [Lock() for _ in nvsmi.get_gpus()]

def getFreeGPU(itrWait : float = 3e1, maxWait : float = 4.32e4):

    global locks
    
    endTime = time.time() + maxWait

    while time.time() < endTime:

        # Try to get a GPU which is free and not used by others
        for device in nvsmi.get_available_gpus():
            if locks[int(device.id)].acquire(timeout=-1):
                return device
          
        time.sleep(itrWait)

    return None

"""
    Runner function used during the training of the models
    It handeles selecting an available CUDA device and waits all are occupied (sometimes buggy)
    It extracts validation metric and also builds the folder structure
"""
def setupRun(
        configFile : Path = Path("config.yaml"),
        runFile : Path = Path("../../../run.py"),
        logFile : Union[Path, None] = None,
        metric : str = "valAcc",
        maximize : bool = True,
        **kwargs
    ):

    global locks

    # Get name
    name = "_".join(["{}={}".format(k, str(v)) for k, v in kwargs.items()])

    # Create folder structure
    root = Path("Experiments").joinpath(name)
    root.mkdir(parents=True)
    updatedConfig = root.joinpath("config.yaml")
    logFile = root.joinpath("Logging") if logFile is None else str(logFile)
    logFile.mkdir(exist_ok=True)

    # Load config
    kwargs.update({"experiment_name": name})
    with updatedConfig.open("a") as out:
        for l in configFile.open().readlines():
            out.write(l.format(**kwargs))

    # Get free device
    device = getFreeGPU()

    # Run
    os.system("export CUDA_VISIBLE_DEVICES={} ; cd {} ; python3 {} --config config.yaml >> out.out 2>&1".format(device.id, str(root), str(runFile)))
    
    # Release lock on device
    locks[int(device.id)].release()

    # Get best result
    tracker = MlflowClient(str(logFile))
    expId = tracker.list_experiments()[0].experiment_id
    runId = tracker.list_run_infos(expId)[0].run_id
    return (max if maximize else min)(x.value for x in tracker.get_metric_history(runId, metric))
