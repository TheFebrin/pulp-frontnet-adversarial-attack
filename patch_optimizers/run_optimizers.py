from typing import Any, Callable, Dict, List
from tinydb import TinyDB, Query
from multiprocessing import Process, Lock
from tqdm import tqdm

from pulp_frontnet.PyTorch.Frontnet.DataProcessor import DataProcessor
from pulp_frontnet.PyTorch.Frontnet.Dataset import Dataset
from pulp_frontnet.PyTorch.Frontnet.Frontnet import FrontnetModel
from pulp_frontnet.PyTorch.Frontnet import Utils
from pulp_frontnet.PyTorch.Frontnet.Utils import ModelManager
from pulp_frontnet.PyTorch.Frontnet.ModelTrainer import ModelTrainer

from patch_optimizers.utils import single_vector_element_cost_f
from patch_optimizers.sliding_window_optimizer import SlidingWindowOptimizer
from patch_optimizers.simulated_annealing_optimizer import SimulatedAnnealingOptimizer
from patch_optimizers.random_optimizer import RandomOptimizer
from patch_optimizers.genetic_optimizer import GeneticOptimizer
from patch_optimizers.optimizer_interface import Optimizer


def create_optimizers_factories(
    model, element_index: int, targeted_attack: bool
) -> List[Callable[[], Optimizer]]:
    def __cost_f_one_element(prediction_with_patch, model_raw_prediction):
        return single_vector_element_cost_f(
            prediction_with_patch=prediction_with_patch,
            model_raw_prediction=model_raw_prediction,
            element_index=element_index,
        )

    def __cost_f_all_elements(prediction_with_patch, model_raw_prediction):
        diffs = [
            (prediction_with_patch[i] - model_raw_prediction[i]) ** 2 for i in range(3)
        ]
        return sum(diffs).detach().numpy().squeeze()

    def __cost_targeted_attack(prediction_with_patch, model_raw_prediction):
        ground_truth = [0, 0, 0]
        diffs = [(prediction_with_patch[i] - ground_truth[i]) ** 2 for i in range(3)]
        # - as we want to get closer to (0, 0, 0)
        return -sum(diffs).detach().numpy().squeeze()

    if targeted_attack:
        print(
            "Using targeted attack cost function. Assuming that ground truth is [0, 0, 0]"
        )
        cost_f = __cost_targeted_attack
    else:
        if element_index == 123:
            cost_f = __cost_f_all_elements
        else:
            cost_f = __cost_f_one_element

    optimizer_factories: List[Callable[[], Optimizer]] = [
        lambda: RandomOptimizer(
            cost_f=cost_f,
            model=model,
            n_dots_to_generate=1000,
            k_dots=1,
            dot_size=10,
        ),
        lambda: SlidingWindowOptimizer(
            cost_f=cost_f,
            model=model,
            k_dots=1,
            dot_size=10,
            stride=5,
        ),
        lambda: SimulatedAnnealingOptimizer(
            x0=(80, 45),
            stride=10,
            max_iters=1000,
            model=model,
            cost_f=cost_f,
            debug=False,
        ),
        lambda: GeneticOptimizer(
            n_iters=20,
            population_size=100,
            model=model,
            cost_f=cost_f,
            max_x=160,
            max_y=96,
            dot_size=10,
            elitism=False,
            debug=False,
        ),
        lambda: GeneticOptimizer(
            n_iters=20,
            population_size=100,
            model=model,
            cost_f=cost_f,
            max_x=160,
            max_y=96,
            dot_size=10,
            elitism=True,
            debug=False,
        ),
    ]
    return optimizer_factories


def create_model():
    model_path = "pulp_frontnet/PyTorch/Models/Frontnet160x32.pt"
    model = FrontnetModel()
    ModelManager.Read(model_path, model)
    model.eval()
    return model


def load_test_set():
    testset_path = "pulp_frontnet/PyTorch/Data/Data/160x96StrangersTestset.pickle"
    [x_test, y_test] = DataProcessor.ProcessTestData(testset_path)
    test_set = Dataset(x_test, y_test)
    return test_set


def run_process(optimizer_factory, test_set, db):
    optimizer = optimizer_factory()
    print("Starting optimizer: ", str(optimizer))
    for i, (x, y) in tqdm(
        enumerate(test_set), desc="Finding patches", total=len(test_set)
    ):
        optimizer = optimizer_factory()
        db_record = {
            "image_idx": i,
        }
        optimal_cost, optimal_patches = optimizer.run(img=x)
        if isinstance(optimal_patches, list):
            optimal_patches = optimal_patches[0]
        optimal_cost = float(optimal_cost)
        db_record[str(optimizer)] = (
            optimal_cost,
            (int(optimal_patches[0]), int(optimal_patches[1])),
        )
        db.insert(db_record)


index_to_db_name = {
    0: "random",
    1: "sliding",
    2: "annealing",
    3: "genetic",
    4: "genetic_elitism",
}

element_index_to_name = {
    0: "x",
    1: "y",
    2: "z",
    123: "xyz",
}


def main() -> None:
    model = create_model()
    test_set = load_test_set()

    targeted_attack = input("Targeted attack? (y/n)")
    targeted_attack = targeted_attack == "y"

    if not targeted_attack:
        element_index = int(
            input("Enter element index to attack: (0=x, 1=y, 2=z, 123=xyz)")
        )
        assert element_index in [0, 1, 2, 123]
    else:
        element_index = None

    optimizer_factories = create_optimizers_factories(
        model=model, element_index=element_index, targeted_attack=targeted_attack
    )
    for i, optimizer_factory in enumerate(optimizer_factories):
        print(f"{i}: {str(optimizer_factory())}")

    factory_idx = int(input("Enter optimizer factory index: "))

    db = TinyDB(
        f"results_dbs/{index_to_db_name[factory_idx]}_{element_index_to_name[element_index]}.json"
    )

    run_process(
        optimizer_factory=optimizer_factories[int(factory_idx)],
        test_set=test_set,
        db=db,
    )


if __name__ == "__main__":
    main()
