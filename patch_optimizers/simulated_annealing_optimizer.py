from typing import List, Tuple
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from patch_optimizers.utils import optimize_for_one_image, apply_path
from patch_optimizers.optimizer_interface import Optimizer


class SimulatedAnnealingOptimizer(Optimizer):
    def __init__(
        self,
        x0: Tuple[int, int],
        model,
        cost_f,
        debug: bool,
        stride: int = 10,
        max_iters: int = 10000,
        alpha: float = 1.0,
    ):
        self._max_iters = max_iters
        self._stride = stride
        self._alpha = alpha
        self._dot_size = 10
        self._stepsize = stride
        self._model = model
        self._cost_f = cost_f
        self._debug = debug
        self.solution = x0
        self.best_solution = x0
        self.cost = 0.0
        self.best_cost = self.cost
        self.cost_history = []
        self.solution_history = []
        self.good_jumps = 0
        self.random_jumps = 0
        assert len(x0) == 2

    def __repr__(self) -> str:
        return type(self).__name__

    def _objective_function(
        self,
        x: int,
        y: int,
        img: torch.Tensor,
        model_raw_prediction: torch.Tensor,
    ) -> float:
        return optimize_for_one_image(
            img=img,
            model_raw_prediction=model_raw_prediction,
            size=self._dot_size,
            x=x,
            y=y,
            model=self._model,
            cost_f=self._cost_f,
        )

    def _random_neighbor(self) -> Tuple[int, int]:
        # img size is 160 x 96
        x, y = self.solution
        x += np.random.randint(-2 * self._stepsize, 2 * self._stepsize)
        x = max(x, 0)
        x = min(x, 160)

        y += np.random.randint(-2 * self._stepsize, 2 * self._stepsize)
        y = max(y, 0)
        y = min(y, 96)
        return (x, y)

    def print_summary(self, img: torch.Tensor) -> None:
        prediction=self._model(img.unsqueeze(0)),
        img_copy = img.clone()
        apply_path(
            img_copy[0].numpy(),
            x=int(self.best_solution[0]),
            y=int(self.best_solution[1]),
            size=self._dot_size,
        )
        cost = -self._cost_f(
            prediction_with_patch=prediction,
            model_raw_prediction=self._model(img.unsqueeze(0)),
        )
        print(f"Steps: {self._max_iters}")
        print(f"Best solution: {self.best_solution}")
        print(f"Best cost: {self.best_cost}")
        print(f"Number of good jumps: {self.good_jumps}")
        print(f"Number of random jumps: {self.random_jumps}")
        print(f"Cost: {cost}")
        raw_pred = self._model(img)
        raw_pred = list(
            map(lambda x: round(float(x.detach().numpy().squeeze()), 4), raw_pred)
        )
        print(f"Prediction on raw: {raw_pred}")
        patch_pred = self._model(img_copy)
        patch_pred = list(
            map(lambda x: round(float(x.detach().numpy().squeeze()), 4), patch_pred)
        )
        print(f"Prediction with patch: {patch_pred}")

        plt.figure(figsize=(15, 5))
        plt.plot(self.cost_history)
        plt.title("Cost history")
        plt.show()

        plt.imshow(img_copy[0].numpy(), cmap="gray")
        plt.title(f"Optimal patch = {self.best_solution}")
        plt.show()

    def run(
        self,
        img: torch.Tensor,
    ) -> Tuple[float, List[Tuple[float, float]]]:
        model_raw_prediction = self._model(img.unsqueeze(0))
        for t in tqdm(
            range(self._max_iters),
            desc="Simulated Annealing",
            position=0,
            disable=not self._debug,
        ):
            x, y = self._random_neighbor()
            new_cost = self._objective_function(
                x=x,
                y=y,
                img=img,
                model_raw_prediction=model_raw_prediction,
            )

            # we minimize the cost function
            if new_cost < self.cost:
                self.cost = new_cost
                self.solution = (x, y)
                self.good_jumps += 1

            elif np.random.rand() < np.exp(
                -self._alpha * (new_cost - self.cost) * t / self._max_iters
            ):
                self.cost = new_cost
                self.solution = (x, y)
                self.random_jumps += 1

            if self.cost < self.best_cost:
                self.best_cost = self.cost
                self.best_solution = self.solution
            self.cost_history.append(self.cost)
            self.solution_history.append(self.solution)

        if self._debug:
            self.print_summary(img=img)
        return -self.best_cost, self.best_solution
