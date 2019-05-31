from bingo.Base.HallOfFame import HallOfFame


class ParetoFront(HallOfFame):

    def __init__(self, secondary_key, primary_key=None,
                 similarity_function=None):
        super().__init__(max_size=None,
                         key_function=primary_key,
                         similarity_function=similarity_function)
        self._key_func_2 = secondary_key

    def update(self, population):
        for indv in population:
            if self._not_dominated(indv) and self._not_similar(indv):
                self._remove_dominated_pf_members(indv)
                self.insert(indv)

    def _not_dominated(self, individual):
        for hof_memeber in self:
            if self._first_dominates(hof_memeber, individual):
                return False
        return True

    def _first_dominates(self, first_indv, second_indv):
        first_keys = (self._key_func(first_indv),
                      self._key_func_2(first_indv))
        second_keys = (self._key_func(second_indv),
                       self._key_func_2(second_indv))
        if first_keys[0] > second_keys[0] or first_keys[1] > second_keys[1]:
            return False

        not_equal = first_keys[0] != second_keys[0] or \
            first_keys[1] != second_keys[1]
        return not_equal

    def _remove_dominated_pf_members(self, individual):
        dominated_hof_members = self._get_dominated_hof_members(individual)
        for i in reversed(dominated_hof_members):
            self.remove(i)

    def _get_dominated_hof_members(self, individual):
        dominated_members = []
        for i, hof_memeber in enumerate(self):
            if self._first_dominates(individual, hof_memeber):
                dominated_members.append(i)
        return dominated_members
