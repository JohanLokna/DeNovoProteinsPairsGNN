# General imports
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from functools import partial
from multiprocessing import Lock
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List

"""
    Runner function for handeling Baysian optimization
    It starts a training procedure and extracts the results
    and logs them - using locks to ensure no race conditions
"""
def runBayesianHP(
    pbounds: dict,
    wrapper,
    nIter : int = 1,
    nParalell : int = 1,
    logPath : Path = Path("./logs.json"),
    fixedPoints : List[dict] = [],
    kind : str = "ucb",
    kappa : float = 100,
    xi : float = 0.1
) -> None:

    assert len(fixedPoints) <= nIter

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
    utility = UtilityFunction(kind=kind, kappa=kappa, xi=xi)

    # Set up lock
    lock = Lock()

    # Helper for running
    def runnerHelper(point = None):

        # Compute next point
        if point is None:
            with lock:
                point = optimizer.suggest(utility)

        # Compute target
        target = wrapper(point)
        
        # Register result
        with lock:
            optimizer.register(
                params=point,
                target=target,
            )

    # Run optimization
    with ThreadPool(processes=nParalell) as pool:
        pool.starmap(runnerHelper, fixedPoints + [() for _ in range(nIter - len(fixedPoints))])


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
