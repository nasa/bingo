import numpy as np

from bingo.selection.selection import Selection
from bingo.util.argument_validation import argument_validation


class BayesianModelSelectionTournament(Selection):

    @argument_validation(tournament_size={">=": 1})
    def __init__(self, tournament_size, logscale=True):
        self._size = tournament_size
        self._logscale = logscale

    def __call__(self, population, target_population_size):
        next_generation = []
        for i in range(target_population_size):
            tournament_members = np.random.choice(population, self._size,
                                                  replace=False)
            winner = self._bayesian_model_selection(tournament_members)
            next_generation.append(winner.copy())
        return next_generation

    def _bayesian_model_selection(self, potential_models):
        marginal_likelihoods = np.array([-model.fitness
                                         for model in potential_models])

        nan_mls = np.isnan(marginal_likelihoods)
        if np.count_nonzero(~nan_mls) == 0:
            return potential_models[0]

        if self._logscale:
            marginal_likelihoods -= np.nanmedian(marginal_likelihoods)
            marginal_likelihoods = np.exp(marginal_likelihoods)

        marginal_likelihoods[nan_mls] = 0
        rand = np.random.random()*sum(marginal_likelihoods)
        index = np.searchsorted(np.cumsum(marginal_likelihoods), rand)
        if np.isnan(potential_models[index].fitness):
            tmp = np.array([-model.fitness for model in potential_models])
            print(tmp)
            print(nan_mls)
            print(np.nanmedian(tmp))
            print(tmp - np.nanmedian(tmp))
            print(marginal_likelihoods)
            print(np.cumsum(marginal_likelihoods))
            print(rand)
            raise RuntimeError
        return potential_models[index]
