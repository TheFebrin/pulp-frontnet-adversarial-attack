from typing import List, Tuple
import torch
import cv2
import numpy as np
from patch_optimizers.utils import apply_path
from patch_optimizers.optimizer_interface import Optimizer


class RandomOptimizer(Optimizer):
    def __init__(
        self,
        cost_f,
        model,
        n_dots_to_generate: int,
        k_dots: int,
        dot_size: int,
    ) -> None:
        self._n_dots_to_generate = n_dots_to_generate
        self._k_dots = k_dots
        self._dot_size = dot_size
        self._cost_f = cost_f
        self._model = model

    def __repr__(self) -> str:
        return type(self).__name__

    def run(
        self,
        img: torch.Tensor,
    ) -> Tuple[float, List[Tuple[float, float]]]:
        """
        img: shape (1, n, m)
        """
        n, m = img[0].shape

        generated_dots = np.hstack(
            (
                np.random.randint(0, n, size=(self._n_dots_to_generate, 1)),
                np.random.randint(0, m, size=(self._n_dots_to_generate, 1)),
            )
        )
        cost_for_each_dot = []
        model_raw_prediction = self._model(img.unsqueeze(0))

        for x, y in generated_dots:
            img_copy = img.clone()
            apply_path(
                img=img_copy[0].numpy(),
                x=x,
                y=y,
                size=self._dot_size,
            )
            cost = -self._cost_f(
                prediction_with_patch=self._model(img_copy.unsqueeze(0)),
                model_raw_prediction=model_raw_prediction,
            )
            cost_for_each_dot.append(cost)

        generated_dots = np.hstack(
            (generated_dots, np.array(cost_for_each_dot).reshape(-1, 1))
        )

        # sort generated_dots by third column
        generated_dots = generated_dots[generated_dots[:, 2].argsort()]

        img_copy = img.clone()
        optimal_patches = []
        for i in range(self._k_dots):
            x, y, cost = generated_dots[i]
            apply_path(
                img=img_copy[0].numpy(),
                x=int(x),
                y=int(y),
                size=self._dot_size,
            )
            optimal_patches.append((int(x), int(y)))

        optimal_cost = self._cost_f(
            prediction_with_patch=self._model(img_copy.unsqueeze(0)),
            model_raw_prediction=model_raw_prediction,
        )
        return optimal_cost, optimal_patches
