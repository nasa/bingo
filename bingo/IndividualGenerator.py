from abc import ABCMeta, abstractmethod


class IndividualGenerator(object, metaclass=ABCMeta):

    @abstractmethod
    def generate(self):
        raise NotImplementedError
