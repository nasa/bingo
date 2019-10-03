import numpy as np

from ...selection.selection import Selection
from ...util.argument_validation import argument_validation


class BayesianModelSelectionTournament(Selection):

    @argument_validation(tournament_size={">=": 1})
    def __init__(self, tournament_size):
        self._size = tournament_size

    def __call__(self, population, target_population_size):
        next_generation = []
        for _ in range(target_population_size):
            tournament_members = np.random.choice(population, self._size,
                                                  replace=False)
            winner = self._bayesian_model_selection(tournament_members)
            next_generation.append(winner.copy())

        return next_generation

    def _bayesian_model_selection(self, potential_models):
        prior = np.full(self._size, 1/self._size)
        evidence = np.array([model.fitness for model in potential_models])
        probabilities = prior * evidence
        probabilities /= np.sum(probabilities)

        index = np.searchsorted(np.cumsum(probabilities), np.random.random())
        return potential_models[index]