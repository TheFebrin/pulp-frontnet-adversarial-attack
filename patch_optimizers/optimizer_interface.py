
import torch
from abc import ABC, abstractmethod
from typing import List, Tuple


class Optimizer(ABC):
    
    @abstractmethod
    def run(
        self,
        img: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> Tuple[float, List[Tuple[float, float]]]:
        pass