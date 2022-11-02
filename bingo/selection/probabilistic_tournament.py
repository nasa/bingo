"""
Probabilistic model selection tournament

An modification of the standard tournament selection process which is
probabilistic and weights the chances for selection by the fitness of the
individuals in the tournament.
"""
import numpy as np

from .selection import Selection
from ..util.argument_validation import argument_validation


class ProbabilisticTournament(Selection):
    """Tournament selection using probabilistic model selection

    Individuals are chosen with a probability equal to the relative vale of 
    their fitness.  When used in conjunction with `NormalizedMarginalLikelihood`
    this results in selection with Bayesian Model Selection (Based on the 
    Fractional Bayes Factor)

    Parameters
    ----------
    tournament_size : int
        size of the tournament
    logscale : bool
        Whether fitnesses of the individuals is in log space. Default True.
    negative : bool
        Whether to invert the fitness of the individual (before log). Default 
        True.
    """

    @argument_validation(tournament_size={">=": 1})
    def __init__(self, tournament_size, logscale=True, negative=False):
        self._size = tournament_size
        self._logscale = logscale
        self._negative = negative

    def __call__(self, population, target_population_size):
        next_generation = []
        for _ in range(target_population_size):
            tournament_members = np.random.choice(
                population, self._size, replace=False
            )
            winner = self._probabilistic_model_selection(tournament_members)
            next_generation.append(winner.copy())
        return next_generation

    def _probabilistic_model_selection(self, potential_models):
        fitnesses = np.array(
            [
                -model.fitness if self._negative else model.fitness
                for model in potential_models
            ]
        )

        nan_fitnesses = np.isnan(fitnesses)
        if np.count_nonzero(~nan_fitnesses) == 0:
            return potential_models[0]

        if self._logscale:
            fitnesses -= np.nanmedian(fitnesses)
            fitnesses = np.exp(fitnesses)

        fitnesses[nan_fitnesses] = 0
        rand = np.random.random() * sum(fitnesses)
        index = np.searchsorted(np.cumsum(fitnesses), rand)
        # if np.isnan(potential_models[index].fitness):
        #     tmp = np.array([-model.fitness for model in potential_models])
        #     print(tmp)
        #     print(nan_fitnesses)
        #     print(np.nanmedian(tmp))
        #     print(tmp - np.nanmedian(tmp))
        #     print(fitnesses)
        #     print(np.cumsum(fitnesses))
        #     print(rand)
        #     raise RuntimeError
        return potential_models[index]
