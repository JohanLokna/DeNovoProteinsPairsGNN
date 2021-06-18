# General imports
from pathlib import Path
import shutil
import os

def runStorkach(hidden_size, N, alpha, lr, 
                modelConfig : Path = Path("config.yaml"), baseConfig : Path = Path("../config.yaml"),
                runFile : Path = Path("../../run.py")):

    # Fix to int
    hidden_size = int(2 ** hidden_size)
    N = int(N)

    # Get name
    name = "model:Strokach_h:{}_N:{}_alpha:{}_lr:{}".format(hidden_size, N, alpha, lr)
    
    # Create folder structure
    Path("Logging").mkdir(exist_ok=True)
    root = Path("Experiments").joinpath(name)
    root.mkdir(parents=True)

    # Create config file
    outfile = root.joinpath("config.yaml")
    shutil.copy(baseConfig, outfile)
    with outfile.open("a") as out:
        for l in modelConfig.open().readlines():
            for k, v in [("hidden_size", hidden_size), ("N", N), ("alpha", alpha), ("lr", lr)]:
                if k + ": " in l:
                    l = l.split(": ")[0] + ": " + str(v)
                    break
            out.write(l)

    # Run
    os.system("cd {} ; python3 {} --config config.yaml".format(str(root), str(runFile)))
