import abc


__all__ = ['GenerativeModel']


class GenerativeModel(abc.ABC):
    @abc.abstractmethod
    def generate(self, message: str, max_len: int):
        pass
