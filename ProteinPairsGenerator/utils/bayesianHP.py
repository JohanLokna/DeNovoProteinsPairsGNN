# General imports
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from multiprocessing import Lock
from multiprocessing.pool import ThreadPool
from pathlib import Path

def runnerHelper(optimizer, utility, objective, lock : Lock):

    # Compute next point
    with lock:
        point = optimizer.suggest(utility)
    
    print(point)

    # Compute target
    target = objective(**point)
    
    # Register result
    with lock:
        optimizer.register(
            params=point,
            target=target,
        )


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

    logger = JSONLogger(path=str(logPath), reset=False)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    # Set up objective
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    # Set up lock
    lock = Lock()

    # Run optimization
    with ThreadPool(processes=nParalell) as pool:
        pool.map(runnerHelper, [(optimizer, utility, wrapper, lock) for _ in range(nIter)])


    # # Run optimization
    # with Pool(processes=nParalell) as pool:

    #     # Run in paralell
    #     points = [optimizer.suggest(utility) for _ in range(nParalell)]
    #     targets = pool.map(wrapper, points)

    #     # Register afterwards
    #     for p, t in zip(points, targets):
    #         optimizer.register(
    #             params=p,
    #             target=t,
    #         )
