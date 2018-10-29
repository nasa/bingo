from abc import ABCMeta, abstractmethod


class GeneGenerator(object, metaclass=ABCMeta):

    @abstractmethod
    def generate(self):
        pass
