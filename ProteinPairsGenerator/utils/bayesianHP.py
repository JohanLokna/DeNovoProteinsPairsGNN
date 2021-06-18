# General imports
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from multiprocessing import Pool
from pathlib import Path

def runBayesianHP(pbounds: dict, wrapper, nIter : int = 1, nParalell : int = 1, logPath : Path = Path("./logs.json")):

    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        verbose=2,
        random_state=42,
    )

    # Load state and set up logger
    if logPath.exists():
      load_logs(optimizer, logs=[str(logPath)])

    logger = JSONLogger(path=str(logPath))
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    # Set up objective
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    # Run optimization
    with Pool(processes=nIter) as pool:

        # Run in paralell
        points = [optimizer.suggest(utility) for _ in range(nParalell)]
        targets = pool.map(wrapper, points)

        # Register afterwards
        for p, t in zip(points, targets):
            optimizer.register(
                params=p,
                target=t,
            )
