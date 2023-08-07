from typing import Tuple, List
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from patch_optimizers.utils import optimize_for_one_image, apply_path


class GeneticOptimizer:
    def __init__(
        self,
        n_iters: int,
        population_size: int,
        max_x: int,
        max_y: int,
        model,
        cost_f,
        dot_size: int,
        elitism: bool,
        chromosome_len: int = 1,
        K=0.9,
        tau0=None,
        tau1=None,
    ):
        """
        Args:
            population_size : int
                Number of cars in the population

            chromosome_len : int
                Length of the list, which is our answer for the problem,
                should be equal to 2 * number of dots. But not used right now.
                We assume only one dot.

            K : int
                Kind of learning rate parameter
        """

        self._model = model
        self._dot_size = dot_size
        self._cost_f = cost_f
        self._n_iters = n_iters
        self._max_x = max_x
        self._max_y = max_y
        self._elitism = elitism
        self.population_size = population_size
        self.K = K
        self.tau0 = tau0
        self.tau1 = tau1

        if self.tau0 is None:
            self.tau0 = K / np.sqrt(2 * np.sqrt(chromosome_len * 2))

        if self.tau1 is None:
            self.tau1 = K / np.sqrt(2 * chromosome_len * 2)

        self.population = np.hstack(
            (
                np.random.randint(0, max_x, size=(self.population_size, 1)),
                np.random.randint(0, max_y, size=(self.population_size, 1)),
            )
        )
        self.sigmas = np.random.uniform(
            low=0, high=0.05, size=(self.population_size, chromosome_len * 2)
        )

        self.cost = np.ones(self.population_size)
        self.cost_history = []
        self.sigmas_history = []
        self.best_sigmas_history = []
        self.population_history = []
        self.best_solution_history = []

    def _parents_selection(self):
        fitness_values = self.cost
        fitness_values = fitness_values - fitness_values.min()
        if fitness_values.sum() > 0:
            fitness_values = fitness_values / fitness_values.sum()
        else:
            fitness_values = np.ones(len(self.population)) / len(self.population)

        ids = np.random.choice(
            np.arange(self.population_size),
            size=self.population_size,
            replace=True,
            p=fitness_values,
        )
        return ids

    def _crossover(self, parent1_id, parent2_id, parents):
        parent1 = parents[parent1_id]
        parent2 = parents[parent2_id]

        return (parent1 + parent2) / 2

    def _mutation(self):
        """ES algorithm based mutation"""
        X = self.population
        Sigmas = self.sigmas

        E = np.random.normal(0, self.tau1, size=Sigmas.shape)
        eps_o = np.random.normal(0, self.tau0)
        Sigmas *= np.exp(E + eps_o)

        self.population = X + np.random.normal(0, 1, size=Sigmas.shape) * Sigmas
        self.population[:, 0] = np.clip(self.population[:, 0], 0, self._max_x)
        self.population[:, 1] = np.clip(self.population[:, 1], 0, self._max_y)
        self.sigmas = Sigmas

    def _mutation2(self):
        """Adding Gaussian Noise"""
        self.population += np.random.normal(0, 0.1, size=self.population.shape)

    def _select_new_population(self):
        best_indi = self.cost.argmax()
        worst_indi = self.cost.argmin()
        best_indi_from_previous_iteration = self.population[best_indi].copy()
        best_sigmas_from_previous_iteration = self.sigmas[best_indi].copy()

        # self.population_history.append(self.population)
        # self.sigmas_history.append(
        #     self.sigmas.mean(axis=0)  # mean of sigmas in population
        # )
        self.cost_history.append((self.cost.min(), self.cost.mean(), self.cost.max()))
        self.best_sigmas_history.append(
            self.population[best_indi]  # sigmas of best individual
        )
        self.best_solution_history.append(self.population[best_indi])

        ids = self._parents_selection()
        parents = self.population[ids]
        parent_sigmas = self.sigmas[ids]

        assert len(self.population) == len(parents) == self.population_size

        children, children_sigmas = [], []

        for _ in range(self.population_size):
            parents_ids = np.random.sample(range(len(parents)), 2)
            child = self._crossover(
                parent1_id=parents_ids[0],
                parent2_id=parents_ids[1],
                parents=parents,
            )
            children.append(child)
            child_sigmas = (
                parent_sigmas[parents_ids[0]] + parent_sigmas[parents_ids[1]]
            ) / 2
            children_sigmas.append(child_sigmas)

        children = np.array(children)
        self.population = children
        self.sigmas = np.array(children_sigmas)

        self._mutation()

        if self._elitism:
            # keep the best individual in the population
            self.population[worst_indi] = best_indi_from_previous_iteration
            self.sigmas[worst_indi] = best_sigmas_from_previous_iteration

    def _objective_function(
        self, x: int, y: int, img: torch.Tensor, ground_truth: torch.Tensor
    ) -> float:
        x = int(x)
        y = int(y)
        img_copy = img.clone()
        apply_path(
            img=img_copy[0][0].numpy(),
            x=x,
            y=y,
            size=self._dot_size,
        )
        return self._cost_f(
            prediction=self._model(img_copy),
            ground_truth=ground_truth,
        )

    def _update_cost_for_population(
        self, img: torch.Tensor, ground_truth: torch.Tensor
    ) -> None:
        for i in range(self.population_size):
            # Note: assume that each individual in population is a 2D point
            self.cost[i] = self._objective_function(
                x=self.population[i][0],
                y=self.population[i][1],
                img=img,
                ground_truth=ground_truth,
            )

    def run(
        self,
        img: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> Tuple[float, List[Tuple[float, float]]]:
        for _ in tqdm(range(self._n_iters), desc="Running genetic optimizer ..."):
            self._update_cost_for_population(
                img=img,
                ground_truth=ground_truth,
            )
            self._select_new_population()

        self._update_cost_for_population(
            img=img,
            ground_truth=ground_truth,
        )

    def best_solution(self):
        return self.population[self.cost.argmax()]

    def plot_cost(self):
        self.cost_history = np.array(self.cost_history)
        plt.figure(figsize=(15, 5))
        plt.plot(self.cost_history)
        maxi_id = self.cost_history[:, 2].argmax()
        maxi_val = self.cost_history[:, 2][maxi_id]
        plt.title(
            f"POPULATION SIZE: {self.population_size} |  BEST_ITER: {maxi_id}  |  MAX: {maxi_val :.3f}"
        )
        plt.legend(["Min", "Mean", "Max"], loc="upper right")

    def plot_sigmas(self, sigmas, mode=""):
        plt.figure(figsize=(15, 5))
        plt.title("Sigmas")
        plt.plot(sigmas)
