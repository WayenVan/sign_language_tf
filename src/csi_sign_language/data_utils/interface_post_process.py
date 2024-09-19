import abc
from typing import List, Tuple

BatchResult = List[List[str]]


class IPostProcess(abc.ABC):
    @abc.abstractmethod
    def process(
        self, hyp: BatchResult, gt: BatchResult
    ) -> Tuple[BatchResult, BatchResult]:
        pass
