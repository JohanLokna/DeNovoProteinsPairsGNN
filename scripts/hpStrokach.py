# General imports
from pathlib import Path
import shutil
import os

def runModel(configFile : Path = Path("config.yaml"), runFile : Path = Path("../run.py"), **kwargs):

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

    # Run
    # os.system("cd {} ; python3 {} --config config.yaml".format(str(root), str(runFile)))
