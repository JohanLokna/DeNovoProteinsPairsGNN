# General imports
from mlflow.tracking import MlflowClient
import nvsmi
import os
from pathlib import Path
import time
from typing import Union

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

def setupRun(
        configFile : Path = Path("config.yaml"),
        runFile : Path = Path("../../../run.py"),
        logFile : Union[Path, None] = None,
        metric : str = "valAcc",
        maximize : bool = True,
        **kwargs
    ):

    # Get name
    name = "_".join(["{}={}".format(k, str(v)) for k, v in kwargs.items()])

    # Create folder structure
    root = Path("Experiments").joinpath(name)
    root.mkdir(parents=True)
    updatedConfig = root.joinpath("config.yaml")
    logFile = root.joinpath("Logging") if logFile is None else str(logFile)

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
    # os.system("export CUDA_VISIBLE_DEVICES={} ; cd {} ; python3 {} --config config.yaml >> out.out".format(device.id, str(root), str(runFile)))
    os.system("cp /mnt/ds3lab-scratch/jlokna/OldStuff/ExpStrokach0.75/Logging -r {}".format(str(root)))

    # Get best result
    tracker = MlflowClient(str(logFile))
    expId = tracker.list_experiments()[0].experiment_id
    runId = tracker.list_run_infos(expId)[0].run_id
    return (max if maximize else min)(float(x) for x in tracker.get_metric_history(runId, metric))

