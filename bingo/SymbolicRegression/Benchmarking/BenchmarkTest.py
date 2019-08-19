
class BenchmarkTest:
    def __init__(self, train_function, score_function):
        self._train_function = train_function
        self._score_function = score_function
        self._best_equation = None
        self._aux_train_info = None

    def train(self, training_data):
        self._best_equation, self._aux_train_info = \
            self._train_function(training_data)

    def score(self, score_data):
        if self._best_equation is None:
            raise RuntimeError("BenchmarkTest must be trained before scoring")
        return self._score_function(self._best_equation, score_data,
                                    self._aux_train_info)