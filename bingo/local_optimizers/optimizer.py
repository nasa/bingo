from abc import ABCMeta, abstractmethod


class OptimizerBase(metaclass=ABCMeta):
    @property
    @abstractmethod
    def objective_fn(self):
        raise NotImplementedError

    @objective_fn.setter
    @abstractmethod
    def objective_fn(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def options(self):
        raise NotImplementedError

    @options.setter
    @abstractmethod
    def options(self, value):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, individual):
        raise NotImplementedError
