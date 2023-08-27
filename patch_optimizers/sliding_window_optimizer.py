from typing import List, Tuple
import torch
import cv2
from patch_optimizers.utils import apply_path
from patch_optimizers.optimizer_interface import Optimizer


class SlidingWindowOptimizer(Optimizer):
    def __init__(
        self,
        cost_f,
        model,
        k_dots: int,
        dot_size: int,
        stride: int,
        find_dots_that_fix_prediction: bool = False,
    ) -> None:
        self._k_dots = k_dots
        self._dot_size = dot_size
        self._stride = stride
        self._cost_f = cost_f
        self._model = model
        self._find_dots_that_fix_prediction = find_dots_that_fix_prediction

    def __repr__(self) -> str:
        return type(self).__name__

    def run(
        self,
        img: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> Tuple[float, List[Tuple[float, float]]]:
        """
        img: shape (1, n, m)
        """
        n, m = img[0].shape
        optimal_patches = []
        optimal_cost = 0
        for _ in range(self._k_dots):
            best_position_so_far = None
            if self._find_dots_that_fix_prediction:
                highest_cost_so_far = 1e18
            else:
                highest_cost_so_far = 0

            for i in range(0, n, self._stride):
                for j in range(0, m, self._stride):
                    img_copy = img.clone()
                    apply_path(
                        img=img_copy[0].numpy(),
                        x=i,
                        y=j,
                        size=self._dot_size,
                    )
                    cost = self._cost_f(
                        prediction=self._model(img_copy.unsqueeze(0)),
                        ground_truth=ground_truth,
                    )
                    if self._find_dots_that_fix_prediction:
                        if (
                            cost <= highest_cost_so_far
                            and (i, j) not in optimal_patches
                        ):
                            highest_cost_so_far = cost
                            best_position_so_far = (i, j)
                    else:
                        if (
                            cost >= highest_cost_so_far
                            and (i, j) not in optimal_patches
                        ):
                            highest_cost_so_far = cost
                            best_position_so_far = (i, j)

            optimal_patches.append(best_position_so_far)
            if self._find_dots_that_fix_prediction:
                optimal_cost = min(optimal_cost, highest_cost_so_far)
            else:
                optimal_cost = max(optimal_cost, highest_cost_so_far)
        return optimal_cost, optimal_patches
