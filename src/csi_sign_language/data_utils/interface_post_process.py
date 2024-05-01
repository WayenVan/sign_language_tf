import abc
from typing import List, Tuple

class IPostProcess(abc.ABC):
    
    abc.abstractmethod
    def process(self, hyp: List[List[str]], gt: List[List[str]]) -> Tuple[any, any]:
        pass
